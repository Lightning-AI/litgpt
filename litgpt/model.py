# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Full definition of a decoder-only transformer-based language model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""

from functools import partial
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Self

from litgpt.attention import (
    DefaultKeysAndValues,
    MultiHeadSelfAttention,
    do_softcapping,
)
from litgpt.config import Config, StartOfLayerHook
from litgpt.kvcache import (
    DenseKVCache,
    KVCache,
    KVCacheParams,
)
from litgpt.scripts.convert_hf_checkpoint import qkv_reassemble
from litgpt.utils import batched_index_select


class GPT(nn.Module):
    def __init__(self, config: Config, **mha_kwargs) -> None:
        """
        Args:
            config: Configuration parameters

        """
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(
            config.n_embd,
            config.padded_vocab_size,
            bias=config.lm_head_bias,
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config, block_idx) for block_idx in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.mha = MultiHeadSelfAttention(config, **mha_kwargs)
        self.max_seq_length = config.block_size
        self._start_of_layer_hook = config.start_of_layer_hook
        # Have dense KV caches been created by `set_kv_caches`?
        self._default_kv_cache = False

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the
        model's context length. This allows setting a smaller number to avoid
        allocating unused memory.

        If KV caches are of type `DenseKVCache`, and they are too small to hold
        `value` entries, a warning message is printed.

        """
        if value > self.config.block_size:
            raise ValueError(
                f"Cannot attend to {value}, block size is only {self.config.block_size}."
                " This is likely because the input text exceeds the supported context length of this model."
            )
        self._max_seq_length = value
        # RoPE cache:
        # `cos`, `sin` of shape `(max_seq_length, config.rope_n_elem)`
        # More precisely, the RoPE cache is recomputed only if
        # `max_seq_length` increases.
        # Note: The RoPE cache is independent of KV caches, since positional
        # encoding is done (on query and key vectors) before the KV cache
        # gets involved (and the KV cache stores encoded key tensors).
        if not hasattr(self, "cos") or self.cos.size(0) < value:
            self.reset_caches()
        # KV caches
        # We do not change them here, but output a warning if default caches are
        # too small
        for l_ix, block in enumerate(self.transformer.h):
            attn = block.attn
            kv_cache = attn.kv_cache
            if kv_cache is not None and isinstance(kv_cache, DenseKVCache) and kv_cache.cache_length < value:
                print(
                    f"KV cache for layer {l_ix} too small: Call 'set_kv_caches(batch_size={kv_cache.batch_size}, max_seq_length={value}) before inference"
                )
                break
        # Multi-head attention
        self.mha.set_seq_length(value, device=self.cos.device)

    def reset_caches(self):
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        else:
            self.cos, self.sin = self.rope_cache(device=self.cos.device)

    def are_kv_caches_assigned(self) -> bool:
        status = [block.attn.kv_cache is not None for block in self.transformer.h]
        result = any(status)
        if result and not all(status):
            raise IndexError("Some layers have KV caches assigned, but not all")
        return result

    def assign_kv_caches(self, kv_caches: List[KVCache]):
        """
        Assigns specific KV caches to the multi-head attention blocks
        of each layer. This can only be done if no caches have been
        assigned or created (see :meth:`set_kv_caches`) before.

        KV caches are required for inference (i.e., calling :meth:`forward` with
        `input_pos` argument). If no KV caches are assigned, inference calls
        fail.

        Args:
            kv_caches: KV caches, one for each layer of the model

        """
        if self.are_kv_caches_assigned():
            raise ValueError("Model has KV caches assigned already")
        if len(kv_caches) != self.config.n_layer:
            raise ValueError(f"kv_caches must have one entry per layer, so {self.config.n_layer} entries")
        batch_size = kv_caches[0].batch_size
        dtype = kv_caches[0].dtype
        for cache, block in zip(kv_caches, self.transformer.h):
            self._check_kv_cache(self.config, cache, batch_size, dtype)
            device = block.attn.device
            if device is not None:
                block.attn.kv_cache = cache.to(device=device)
            else:
                block.attn.kv_cache = cache

    def set_kv_caches(
        self,
        batch_size: int,
        dtype: Optional[torch.dtype] = None,
        max_seq_length: Optional[int] = None,
    ):
        """
        This method can be called only if KV caches have not been assigned
        with :meth:`assign_kv_caches`. It creates default (dense) KV caches
        for every layer. These may require a lot of memory. If this is an
        issue, consider :meth:`assign_kv_caches` with KV caches of restricted
        size.

        KV caches are required for inference (i.e., calling :meth:`forward` with
        `input_pos` argument). If no KV caches are assigned, inference calls
        fail.

        Args:
            batch_size: Inference batch size
            dtype: Data type for buffers
            max_seq_length: Cache length. If not given, we use
                `self.max_seq_length`

        """
        if self.are_kv_caches_assigned() and not self._default_kv_cache:
            raise ValueError("Model has KV caches assigned already")
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        for block in self.transformer.h:
            attn = block.attn
            device = attn.device
            kv_cache = attn.kv_cache
            if (
                kv_cache is None
                or kv_cache.batch_size != batch_size
                or kv_cache.cache_length != max_seq_length
                or kv_cache.device != device
                or kv_cache.dtype != dtype
            ):
                if kv_cache is not None:
                    device = kv_cache.device if device is None else device
                    dtype = kv_cache.dtype if dtype is None else dtype
                attn.create_default_kv_cache(
                    batch_size=batch_size,
                    device=device,
                    dtype=dtype,
                    max_sequence_length=max_seq_length,
                )
        self._default_kv_cache = True

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache(device=self.cos.device)
        self.mha.set_seq_length(self.max_seq_length, device=self.cos.device)

    def set_start_of_layer_hook(
        self,
        hook: Optional[StartOfLayerHook],
    ):
        """
        Sets a function `hook(x, block_idx, input_pos)`, which is called
        in :meth:`forward` at the start of each layer. Here, `x` is the
        layer input, `block_idx` the number of the layer, and `input_pos`
        the position in the sequence. The hook is called with the output
        of the final layer (input of head model), where
        `block_idx=self.config.n_layer`.

        The default start of layer hook is `self.config.start_of_layer_hook`.
        This is overwritten here.

        Args:
            hook: Hook function to be set, or `None` to remove hook

        """
        self._start_of_layer_hook = hook

    @staticmethod
    def _check_kv_cache(
        config: Config,
        kv_cache: KVCache,
        batch_size: int,
        dtype: torch.dtype,
    ):
        params = kv_cache.get_params()
        if config.n_query_groups != params.n_query_groups:
            raise ValueError(
                f"config and kv_cache not compatible: config.n_query_groups = {config.n_query_groups} != {params.n_query_groups} = kv_cache.n_query_groups"
            )
        if config.n_head != params.n_head:
            raise ValueError(
                f"config and kv_cache not compatible: config.n_head = {config.n_head} != {params.n_head} = kv_cache.n_head"
            )
        head_size = config.head_size
        if head_size != params.head_size:
            raise ValueError(
                f"config and kv_cache not compatible: config.head_size = {head_size} != {params.head_size} = kv_cache.head_size"
            )
        if batch_size != params.batch_size:
            raise ValueError(f"kv_cache.batch_size = {params.batch_size}, must be {batch_size}")
        if dtype != params.dtype:
            raise ValueError(f"kv_cache.dtype = {params.dtype}, must be {dtype}")

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        input_pos: Optional[int] = None,
        lm_head_chunk_size: int = 0,
        skip_lm_head: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        There are two different contexts in which this method is called:

        - Training: `input_pos` not given. KV cache is not needed.
        - Inference, `input_pos` is given. There are two cases: `input_pos=0`
          (prefill) and `input_pos > 0` (generation). For prefill, KV caches
          must have been assigned (:meth:`assign_kv_caches` or
          :meth:`set_kv_caches`). We must have
          `T <= model.kv_cache_max_prefill_length()`.
        - For generation, KV caches must have been assigned
          (:meth:`assign_kv_caches` or :meth:`set_kv_caches`). We check that
          `input_pos == kv_cache.next_token_pos`. Note that `T > 1` is
          permitted here as well.

        Note: If this method is called with `input_pos=0` (prefill) after
        generation calls, a new inference sequence is started. The batch
        size for the new sequence can be different.

        Token generation (`input_pos > 0`) and `T > 1`:

        This situation is non-standard, since `idx` needs to provide tokens at
        positions `input_pos:(input_pos + T)`, whereas the logits are for
        generating tokens at `(input_pos + 1):(input_pos + T + 1)`, so only the
        last position is needed to generate a new token. Use cases:
        - Updating KV caches sequentially if prompt size is larger than max
          prefill length of cache
        - Speculative decoding. Here, `idx` comes from the cheaper proposal
          model, and the logits are needed for the accept/reject probabilities.

        Args:
            idx: Token indices of input sequences, shape `(B, T)`, where `B`
                is batch size.
            input_pos: See above. Defaults to `None`
            lm_head_chunk_size: Optional. If `lm_head_chunk_size > 0`, the final
                `lm_head` computation is done in chunks of this size.
            skip_lm_head: If `True`, we do not apply the final LM head
                `self.lm_head`.

        Returns:
            Logit outputs, shape `(B, T, config.padded_vocab_size)`. If
            `lm_head_chunk_size > 0`, this is a list of chunks of shape
            `(B, lm_head_chunk_size, config.padded_vocab_size)`, the final
            entry can be shorter.
            If `skip_lm_head` is `True`, we return the final layer outputs,
            shape `(B, T, config.n_embd)`.

        """
        if idx.ndim == 1:
            idx = idx.unsqueeze(0)
        elif idx.ndim != 2:
            raise ValueError(f"idx must be 1D or 2D tensor, but idx.shape = {idx.shape}")
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")
        for_prefill = False
        if input_pos is not None:
            # Few tokens generation. This needs a KV cache. If none is assigned,
            # the call fails
            if not self.are_kv_caches_assigned():
                raise ValueError(
                    "KV caches are not assigned. Assign KV caches with 'assign_kv_caches' or create default caches with 'set_kv_caches'"
                )
            for_prefill = input_pos == 0
            if not for_prefill:
                for block_idx, block in enumerate(self.transformer.h):
                    kv_cache = block.attn.kv_cache
                    if kv_cache.next_token_pos is None:
                        raise ValueError("Inference calls need to start with pre-fill, i.e. 'input_pos=0'")
                    if kv_cache.next_token_pos != input_pos:
                        raise ValueError(
                            f"KV cache for layer {block_idx}: input_pos = {input_pos} != {kv_cache.next_token_pos} = kv_cache.next_token_pos"
                        )
                    if kv_cache.max_tokens_forward < T:
                        raise ValueError(
                            f"KV cache for layer {block_idx}: T = {T}, must be <= max_tokens_forward = {kv_cache.max_tokens_forward}"
                        )

            if self.config.rope_n_elem > 0:
                input_pos_array = torch.arange(input_pos, input_pos + T, device=self.cos.device, dtype=torch.int64)
                cos = batched_index_select(self.cos, 0, input_pos_array).unsqueeze(0)
                sin = batched_index_select(self.sin, 0, input_pos_array).unsqueeze(0)
            else:
                cos = sin = None
        else:
            # Unsqueeze to have a batch dimension
            cos = self.cos[:T].unsqueeze(0)
            sin = self.sin[:T].unsqueeze(0)
        # `cos`, `sin` have shape `(1, T, config.rope_n_elem)`, or shape
        # `(1, T, config.rope_n_elem, 2)`

        x = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        if self.config.scale_embeddings:
            x = x * torch.tensor(self.config.n_embd**0.5, dtype=x.dtype)

        hook = self._start_of_layer_hook
        for block_idx, block in enumerate(self.transformer.h):
            if for_prefill:
                # Complain if batch size of cache is too small
                eff_batch_size = x.shape[0]
                attn = block.attn
                if attn.kv_cache.batch_size < eff_batch_size:
                    raise ValueError(
                        f"Batch size {eff_batch_size} is too large for KV cache layer {block_idx} (batch size {attn.kv_cache.batch_size}). Use 'assign_kv_caches' or `set_kv_caches'"
                    )
            if hook is not None:
                # Call start of layer hook, passing detached layer input
                hook(x.detach(), block_idx, input_pos)
            if self.config.rope_indices is not None:
                # Select global (0) or local (1) variant
                _cos = cos[..., self.config.rope_indices[block_idx]]
                _sin = sin[..., self.config.rope_indices[block_idx]]
            else:
                _cos = cos
                _sin = sin
            x = block(x, _cos, _sin, idx, self.mha, input_pos)

        if hook is not None:
            # Hook is also called for the input to the head block
            hook(x.detach(), self.config.n_layer, input_pos)
        x = self.transformer.ln_f(x)
        if skip_lm_head:
            return x
        clamp_head = partial(do_softcapping, thresh=self.config.final_logit_softcapping)
        if lm_head_chunk_size > 0:
            # chunk the lm head logits to reduce the peak memory used by autograd
            return [clamp_head(self.lm_head(x_i)) for x_i in x.split(lm_head_chunk_size, dim=1)]
        else:
            return clamp_head(self.lm_head(x))  # (B, T, padded_vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def rope_cache(
        self,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recomputes the RoPE cache, consisting of tensors `cos`, `sin`.

        Args:
            device: Device for RoPE cache tensors

        Returns:
            `(cos, sin)`, each of shape `(max_seq_length, config.rope_n_elem)`
            or of shape `(max_seq_length, config.rope_n_elem, 2)`.

        """
        if self.config.rope_adjustments is None:
            extra_config = None
        else:
            adjusted_params_required = ["factor", "low_freq_factor", "high_freq_factor", "original_max_seq_len"]
            params_present = [param in self.config.rope_adjustments for param in adjusted_params_required]
            num_params_present = sum(params_present)

            if num_params_present == 0:
                extra_config = None  # uses standard RoPE
            elif num_params_present == 4:
                # These parameters should always be used together so that we don't interfere with standard rope
                extra_config = {name: self.config.rope_adjustments[name] for name in adjusted_params_required}
            elif "factor" in self.config.rope_adjustments:
                # linear RoPE
                adjusted_params_required = ["factor"]
                extra_config = {name: self.config.rope_adjustments[name] for name in adjusted_params_required}
            else:
                # Some but not all parameters are specified; raise an error
                missing_params = [
                    param for param, present in zip(adjusted_params_required, params_present) if not present
                ]
                raise ValueError(
                    f"The following adjusted RoPE parameters are missing in rope_adjustments: {', '.join(missing_params)}. "
                    "All adjusted RoPE parameters must be specified together."
                )

        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
            extra_config=extra_config,
            rope_local_base_freq=self.config.rope_local_base_freq,
        )

    def clear_kv_caches(self) -> None:
        """
        Note that KV cache objects are removed only if they have not been
        assigned with :meth:`assign_kv_caches`.

        """
        if self._default_kv_cache:
            for block in self.transformer.h:
                block.attn.kv_cache = None
            self._default_kv_cache = False

    def get_kv_cache_params(self, layer_idx: int) -> Optional[KVCacheParams]:
        """
        Args:
            layer_idx: Layer for which KV cache params are requested

        Returns:
            Parameters for KV caches (see above), or `None` if KV caches are
            not assigned.

        """
        if not (0 <= layer_idx < self.config.n_layer):
            raise IndexError(f"layer_idx={layer_idx}, must be in [0, {self.config.n_layer})")
        kv_cache = self.transformer.h[layer_idx].attn.kv_cache
        return None if kv_cache is None else kv_cache.get_params()

    def kv_cache_max_tokens_forward(self) -> Optional[int]:
        """
        Returns:
            Smallest `max_tokens_forward` over all KV caches, or `None` if KV
            caches are not assigned.

        """
        caches = [layer.attn.kv_cache for layer in self.transformer.h]
        if any(cache is None for cache in caches):
            return None
        else:
            return min(kvc.max_tokens_forward for kvc in caches)

    def kv_cache_max_prefill_length(self) -> Optional[int]:
        """
        Returns:
            Smallest `max_prefill_length` over all KV caches, or `None` if KV
            caches are not assigned or if `max_prefill_length is None` for all
            KV caches.

        """
        caches = [layer.attn.kv_cache for layer in self.transformer.h]
        if any(cache is None for cache in caches):
            return None
        else:
            mlps = [kvc.max_prefill_length for kvc in caches]
            if all(mlp is None for mlp in mlps):
                return None
            else:
                return min(mlp for mlp in mlps if mlp is not None)


class Block(nn.Module):
    def __init__(
        self,
        config: Config,
        block_idx: int,
        kv_cache: Optional[KVCache] = None,
    ) -> None:
        super().__init__()
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )

        self.norm_1 = nn.Identity() if not config.norm_1 else config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, block_idx, kv_cache=kv_cache)
        self.post_attention_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps) if config.post_attention_norm else nn.Identity()
        )
        self.norm_2 = (
            nn.Identity()
            if not config.norm_2
            else (None if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps))
        )
        self.mlp = config.mlp_class(config)
        self.post_mlp_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps) if config.post_mlp_norm else nn.Identity()
        )
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        token_idx: torch.Tensor,
        mha: MultiHeadSelfAttention,
        input_pos: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Non-parallel residual       Parallel residual
           ┌─ x                     ┌─ x ──────────────────┐             Note: if `shared_attention_norm` is True,
           │  ↓                     │  ↓                   ↓                   the output from `norm_1` is reused
           │  norm_1                │  norm_1  ───────►    norm_2
           │  ↓                     │  ↓                   ↓
           │  attn                  │  attn                MLP
           │  ↓                     │  ↓                   ↓
           |  post_attn_norm        |  post_attn_norm      post_mlp_norm
           |  ↓                     |  ↓                   ↓
        ┌─ └► +                     └► + ◄─────────────────┘
        |     ↓
        │     norm_2
        │     ↓
        │     MLP
        │     ↓
        |     post_mlp_norm
        |     ↓
        └───► +
        """

        x_normed = self.norm_1(x)
        attention_output = self.attn(
            x_normed,
            cos=cos,
            sin=sin,
            token_idx=token_idx,
            mha=mha,
            input_pos=input_pos,
        )
        attention_output = self.post_attention_norm(attention_output)

        if self.config.parallel_residual:
            if not self.config.shared_attention_norm:
                x_normed = self.norm_2(x)
            x = attention_output + x
        else:
            x = attention_output + x
            x_normed = self.norm_2(x)

        return self.post_mlp_norm(self.mlp(x_normed)) + x


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        config: Config,
        block_idx: int,
        kv_cache: Optional[KVCache] = None,
    ) -> None:
        super().__init__()
        # key, query and value projections for all heads, but in a batch
        self.qkv = nn.Linear(
            config.n_embd,
            (config.n_head + 2 * config.n_query_groups) * config.head_size,  # support for grouped/multi queries
            bias=config.bias or config.attn_bias,
        )
        # output projection
        self.proj = nn.Linear(config.head_size * config.n_head, config.n_embd, bias=config.bias)
        # KV cache (needed for inference)
        self.kv_cache = kv_cache

        if config.norm_qk:
            norm_q_size = config.n_head * config.head_size if config.norm_qk_type == "olmo2" else config.head_size
            norm_k_size = (
                config.n_query_groups * config.head_size if config.norm_qk_type == "olmo2" else config.head_size
            )
            self.norm_q = config.norm_class(norm_q_size, eps=config.norm_eps)
            self.norm_k = config.norm_class(norm_k_size, eps=config.norm_eps)
        else:
            self.norm_q = self.norm_k = None

        self.config = config
        self.block_idx = block_idx

    @property
    def device(self) -> Optional[torch.device]:
        w = self.qkv.weight
        return None if w is None else w.device

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        token_idx: torch.Tensor,
        mha: MultiHeadSelfAttention,
        input_pos: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            cos: RoPE parameters
            sin: RoPE parameters
            token_idx: Token indexes corresponding to `x`
            mha: Multi-head self-attention code
            input_pos: See :meth:`GPT.forward`

        Returns:
            Output tensor
        """
        # Notation:
        # - B          | batch size
        # - T          | time-step (sequence length)
        # - C          | model's embeddings size (n_embd)
        # - C*         | attentions's embeddings size
        # - hs         | head size
        # - nh_(q,k,v) | number of heads for query, key and value
        # - n_query_groups = nh_k = nh_v | number of query groups sharing key and value heads
        # alternative notation: num_kv_groups = n_query_groups
        # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
        # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        #   │    │    │    │         │        │                 │
        # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
        # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
        # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
        # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
        # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
        # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
        #         MHA                    GQA                   MQA
        #   n_query_groups=4       n_query_groups=2      n_query_groups=1
        #
        # credit https://arxiv.org/pdf/2305.13245.pdf
        head_size = self.config.head_size
        n_head = self.config.n_head
        n_query_groups = self.config.n_query_groups
        rope_n_elem = self.config.rope_n_elem
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        if input_pos is not None:
            for_prefill = input_pos == 0
            if self.kv_cache is None:
                raise ValueError(
                    "KV caches are not assigned. Assign KV caches with 'assign_kv_caches' or create default caches with 'set_kv_caches'"
                )
            if not for_prefill:
                if self.kv_cache.next_token_pos is None:
                    raise ValueError("Inference calls need to start with pre-fill, i.e. 'input_pos=0'")
                if self.kv_cache.next_token_pos != input_pos:
                    raise ValueError(
                        f"KV cache: input_pos = {input_pos} != {self.kv_cache.next_token_pos} = kv_cache.next_token_pos"
                    )
                if self.kv_cache.max_tokens_forward < T:
                    raise ValueError(
                        f"KV cache: T = {T}, must be <= max_tokens_forward = {self.kv_cache.max_tokens_forward}"
                    )

        # Perform a single multiplication operation using a combined QKV matrix to calculate `query`, `key`, and `value`
        # instead of individually multiplying the input `x` with the respective weight matrices.
        qkv = self.qkv(x)  # (B, T, 3xC*)

        # Define query, key and value sizes.
        # If grouped/multi query is enabled, these sizes are not equal (see the diagram above).
        query_size = n_head * head_size
        key_size = value_size = n_query_groups * head_size
        # Split qkv into query, key and value matrices.
        q, k, v = qkv.split((query_size, key_size, value_size), dim=-1)  # 3x(B, T, C*)

        if self.config.norm_qk and self.config.norm_qk_type == "olmo2":
            q = self.norm_q(q)
            k = self.norm_k(k)

        # To place the num_heads (nh) dimension right after the batch (B) dimension, the first step is to decouple the
        # embedding size (C) into num_heads (nh) and head_size (hs).

        # The original GQA paper is followed here and the term query groups is used.
        # alternative notation: Query groups are also referred to as KV groups.
        q = q.view(B, T, n_head, head_size)  # (B, T, nh_q, hs)
        k = k.view(B, T, n_query_groups, head_size)  # (B, T, nh_k, hs)
        v = v.view(B, T, n_query_groups, head_size)  # (B, T, nh_k, hs)

        # The tensors `query`, `key`, and `value` are now accurately structured: within each batch element (B), there are
        # multiple heads (nh_q), and within each head, there is a sequence of elements (T), each represented by a vector
        # of size `hs`.
        # Note that `nh_k` can be smaller than `nh_q` (but the latter must be a
        # multiple of the former). This works with the
        # `scaled_dot_product_attention` implementations below.
        q = q.transpose(1, 2)  # (B, nh_q, T, hs)
        k = k.transpose(1, 2)  # (B, nh_k, T, hs)
        v = v.transpose(1, 2)  # (B, nh_k, T, hs)

        if self.config.norm_qk and self.config.norm_qk_type == "default":
            q = self.norm_q(q)
            k = self.norm_k(k)

        # Unlike standard positional embeddings rotary embeddings must be applied at every layer.
        if rope_n_elem > 0:
            q_roped = apply_rope(q[..., :rope_n_elem], cos, sin)
            k_roped = apply_rope(k[..., :rope_n_elem], cos, sin)
            q = torch.cat((q_roped, q[..., rope_n_elem:]), dim=-1)  # (B, nh_q, T, hs)
            k = torch.cat((k_roped, k[..., rope_n_elem:]), dim=-1)  # (B, nh_k, T, hs)

        # Inner part of multi-head self-attention computation
        if input_pos is None:
            # Default causal self-attention
            y, _ = mha(
                query=q,
                k_and_v=DefaultKeysAndValues(k, v),
                block_idx=self.block_idx,
            )
        else:
            # Defer this to KV cache
            y = self.kv_cache(
                query=q,
                key=k,
                value=v,
                token_idx=token_idx,
                input_pos=input_pos,
            )

        # Output projection
        y = self._transform_output(y, query=q, mha=mha)
        return self.proj(y)  # (B, T, C)

    def _transform_output(
        self,
        y: torch.Tensor,
        query: torch.Tensor,
        mha: MultiHeadSelfAttention,
    ) -> torch.Tensor:
        return y

    def create_default_kv_cache(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: Optional[int] = None,
    ):
        self.kv_cache = DenseKVCache(
            config=self.config,
            batch_size=batch_size,
            block_idx=self.block_idx,
            device=device,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
        )

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with legacy checkpoints."""

        for attr in ("weight", "bias"):
            legacy_key = f"{prefix}attn.{attr}"
            current_key = f"{prefix}qkv.{attr}"
            if legacy_key in state_dict:
                state_dict[current_key] = qkv_reassemble(state_dict.pop(legacy_key), self.config)

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config, intermediate_size: Optional[int] = None) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.fc = nn.Linear(config.n_embd, self.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(self.intermediate_size, config.n_embd, bias=config.bias)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x, approximate=self.config.gelu_approximate)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config, intermediate_size: Optional[int] = None) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.fc_1 = nn.Linear(config.n_embd, self.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, self.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(self.intermediate_size, config.n_embd, bias=config.bias)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = F.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class GemmaMLP(LLaMAMLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = F.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
        return self.proj(x)


class LLaMAMoE(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(
            LLaMAMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.n_expert)
        )
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derived from: https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
        See also figure 1 in https://arxiv.org/abs/2211.15841
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        x = x.view(-1, C)  # (B*T, C)
        router = self.gate(x)  # (B*T, n_expert)
        probs, indices = torch.topk(router, self.config.n_expert_per_token)  # (B*T, n_expert_per_token)
        probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
        masks = indices.unsqueeze(-1) == torch.arange(self.config.n_expert, device=x.device)
        masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)
        y = torch.zeros_like(x)  # (B*T, C)
        for mask, expert in zip(masks, self.experts):
            token_idx, expert_idx = torch.where(mask)
            y[token_idx] += probs[token_idx, expert_idx, None] * expert(x[token_idx])
        return y.view(B, T, C)


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
    extra_config: Optional[dict] = None,
    rope_local_base_freq: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Enhanced Transformer with Rotary Position Embedding.

    Args:
        seq_len (int): Sequence length.
        n_elem (int): Number of elements (head dimension).
        device (torch.device, optional): Device for tensor allocations.
        base (int, optional): Base for computing inverse frequencies.
        condense_ratio (int, optional): Ratio to condense the position indices.
        extra_config (dict, optional): Configuration parameters for
            frequency adjustments (used by Llama 3.1 and 3.2)
        rope_local_base_freq: If given, this is an alternative value for
            `base`. In this case, the returned tensors have an extra dimension.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Cosine and sine caches for RoPE.
            Shapes are `(seq_len, n_elem)` if `rope_local_base_freq` is not
            given, otherwise `(seq_len, n_elem, 2)`, so that `[..., 0]` is for
            `base`, and `[..., 1]` for `rope_local_base_freq`.

    """

    # Compute the inverse frequencies theta
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    if extra_config is not None:
        factor = extra_config["factor"]
        if "original_max_seq_len" in extra_config:
            orig_context_len = extra_config["original_max_seq_len"]
            low_freq_factor = extra_config["low_freq_factor"]
            high_freq_factor = extra_config["high_freq_factor"]

            wavelen = 2 * torch.pi / theta
            ratio = orig_context_len / wavelen
            smooth_factor = (ratio - low_freq_factor) / (high_freq_factor - low_freq_factor)
            smooth_factor = torch.clamp(smooth_factor, min=0.0, max=1.0)

            # Compute adjusted_theta without masked indexing
            adjusted_theta = (1 - smooth_factor) * (theta / factor) + smooth_factor * theta
            theta = adjusted_theta
        else:
            theta = theta / factor

    # Create position indices `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)
    # If `n_elem` is odd, the final dimension of `idx_theta` has size
    # `n_elem + 1`, so need to cut something off.

    # Due to a current bug in Hugging Face, in the case `n_elem == 1`, we leave
    # `idx_theta`, `cos`, `sin` as is. Things work out in `apply_rope` due to
    # broadcasting. If we shorten `idx_theta`, unit tests comparing to
    # Hugging Face fail.
    # https://github.com/huggingface/transformers/issues/35233
    # TODO: Remove `> 1` once HF bug is fixed!
    if idx_theta.shape[-1] > n_elem > 1:
        idx_theta = idx_theta[..., :n_elem]

    # if rope_local_base_freq is given, have a separate rope value for local embedding
    # For now, we use default RoPE for local embedding
    if rope_local_base_freq is not None:
        local_theta = 1.0 / (rope_local_base_freq ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
        local_idx_theta = torch.outer(seq_idx, local_theta)
        local_idx_theta = local_idx_theta.repeat(1, 2)
        # TODO: Remove `> 1` once HF bug is fixed!
        if local_idx_theta.shape[-1] > n_elem > 1:
            local_idx_theta = local_idx_theta[..., :n_elem]
        idx_theta = torch.stack((idx_theta, local_idx_theta), dim=-1)

    return torch.cos(idx_theta), torch.sin(idx_theta)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies RoPE transform to `x`. Note that `cos`, `sin` need to have a batch
    dimension.

    Args:
        x: Input tensor, `(B, ..., T, head_size)`
        cos: Cached cosines, `(B, T, head_size)` or `(1, T, head_size)`
        sin: Cached sines, `(B, T, head_size)` or `(1, T, head_size)`

    Returns:
        Encoded tensor, `(B, ..., T, head_size)`
    """
    if cos.ndim != 3:
        raise ValueError(f"cos must be three-dimensional, but shape is {cos.shape}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos, sin must have same shape, but cos.shape={cos.shape}, sin.shape={sin.shape}")
    head_size_half = x.size(-1) // 2
    x1 = x[..., :head_size_half]  # (B, ..., T, head_size/2)
    x2 = x[..., head_size_half:]  # (B, ..., T, head_size/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, ..., T, head_size)
    dims_diff = x.ndim - cos.ndim
    if dims_diff > 0:
        # Ensure that shapes of `x`, `cos`, `sin` align
        new_shape = cos.shape[0:1] + (1,) * dims_diff + cos.shape[1:]
        cos = cos.view(*new_shape)
        sin = sin.view(*new_shape)

    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-6, add_unit_offset: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        weight = (1 + self.weight) if self.add_unit_offset else self.weight
        return (x_normed * weight.float()).to(dtype=dtype)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
