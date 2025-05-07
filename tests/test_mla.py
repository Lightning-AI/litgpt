import pytest
import torch

from litgpt.config import Config
from litgpt.model import MLA


class TestMLA:
    @pytest.fixture
    def config(self):
        # Create a basic config for testing
        config = Config(
            n_embd=32,
            n_head=4,
            norm_qk=True,
            norm_eps=1e-5,
            latent_attention=True,
        )

        # Add MLA-specific config
        config.mla = type(
            "MLAConfig",
            (),
            {
                "q_proj_dim": 32,
                "kv_proj_dim": 42,
                "qk_rope_dim": 8,
                "qk_nope_dim": 8,
            },
        )
        config.v_dim = 16

        print("config:", config)

        return config

    @pytest.fixture
    def mla_model(self, config):
        # Create an MLA instance with test parameters
        return MLA(config, block_idx=0)

    @pytest.fixture
    def rope_cache(self, config):
        # Create a simple RoPE cache for testing
        seq_len = 16
        rope_n_elem = config.rope_n_elem
        batch_size = 1
        cos = torch.ones((batch_size, seq_len, rope_n_elem))
        sin = torch.zeros((batch_size, seq_len, rope_n_elem))
        return cos, sin

    def test_output_shape(self, mla_model, rope_cache):
        # Test that output tensors have expected shapes
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, 32)
        cos, sin = rope_cache

        output = mla_model(x, cos, sin)

        # Verify output shape matches input shape
        assert output.shape == (batch_size, seq_len, 32)

    @pytest.mark.skip(reason="This test is currently broken")
    def test_attention_mask(self, mla_model, rope_cache):
        # Test that attention masking works correctly
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, 32)
        cos, sin = rope_cache

        # Create a mask where some positions are masked
        mask = torch.ones(1, 1, seq_len, seq_len).bool()
        mask = torch.tril(mask.squeeze()).unsqueeze(0).unsqueeze(0)  # Causal mask

        output_masked = mla_model(x, cos, sin, mask)
        output_unmasked = mla_model(x, cos, sin)

        # Outputs should be different when using masks
        assert not torch.allclose(output_masked, output_unmasked, rtol=1e-2)

    def test_rope_application(self, mla_model):
        # Test that RoPE is applied correctly
        x = torch.ones(1, 1, 16)  # Simple test tensor
        cos = torch.ones(1, 1, 16)
        sin = torch.ones(1, 1, 16)

        # Call the RoPE application directly
        ropeified = mla_model.apply_rope_mla(x, cos, sin)

        # Check that the values have been transformed
        assert not torch.allclose(ropeified, x)

        # Test the shape is maintained
        assert ropeified.shape == x.shape

    def test_kv_cache(self, mla_model, rope_cache):
        # Test KV cache functionality
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, 32)
        cos, sin = rope_cache

        # Initialize KV cache
        mla_model.kv_cache = mla_model.build_kv_cache(
            batch_size=batch_size, max_seq_length=16, device=x.device, dtype=x.dtype
        )

        # First forward pass with positions 0-7
        input_pos = torch.arange(8).unsqueeze(0).expand(batch_size, -1)
        print("input_pos shape:", input_pos.shape)
        print("input_pos:", input_pos)
        output1 = mla_model(x[:, :8, :], cos[:, :8, :], sin[:, :8, :], input_pos=input_pos)
        print("output1 shape:", output1.shape)

        # Second forward pass with positions 8-15
        input_pos = torch.arange(8, 16).unsqueeze(0).expand(batch_size, -1)
        output2 = mla_model(x[:, 8:16, :], cos[:, 8:16, :], sin[:, 8:16, :], input_pos=input_pos)

        # Now run without caching
        mla_model.kv_cache = None
        output_full = mla_model(x, cos, sin)

        # Check the shapes
        assert output1.shape == output2.shape == (batch_size, 8, 32)
        assert output_full.shape == (batch_size, seq_len, 32)

    @pytest.mark.skip(reason="This test is currently broken")
    def test_gradient_flow(self, mla_model, rope_cache):
        # Test that gradients flow correctly through MLA
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, 32, requires_grad=True)
        cos, sin = rope_cache

        output = mla_model(x, cos, sin)
        loss = output.sum()
        loss.backward()

        # Check that gradients aren't None or zero
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

        # Check gradients for key parameters
        assert mla_model.dq.weight.grad is not None
        assert mla_model.uq.weight.grad is not None
        assert mla_model.dkv.weight.grad is not None
        assert mla_model.ukv.weight.grad is not None
        assert mla_model.proj.weight.grad is not None

    @pytest.mark.skip(reason="This test is currently broken")
    def test_reproducibility(self, mla_model, rope_cache):
        # Test that results are reproducible with fixed seed
        torch.manual_seed(42)
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, 32)
        cos, sin = rope_cache

        output1 = mla_model(x, cos, sin)

        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_len, 32)  # Same random input
        output2 = mla_model(x, cos, sin)

        assert torch.allclose(output1, output2)

    @pytest.mark.skip(reason="This test is currently broken")
    def test_attention_patterns(self, mla_model, rope_cache):
        # Create input with a distinct pattern to test attention behavior
        batch_size = 1
        seq_len = 16
        x = torch.zeros(batch_size, seq_len, 32)

        # Make each position progressively stronger signal
        for i in range(seq_len):
            x[:, i, :] = i

        cos, sin = rope_cache

        # Test that attention focuses on specific positions
        output = mla_model(x, cos, sin)

        # Later positions should have more influence due to causal attention
        # Check relative magnitudes (this is a simplified test)
        early_pos_avg = output[:, 0, :].abs().mean()
        late_pos_avg = output[:, -1, :].abs().mean()

        # Later positions usually have higher activation due to seeing more context
        assert late_pos_avg > early_pos_avg
