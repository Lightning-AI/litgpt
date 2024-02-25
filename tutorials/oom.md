## Dealing with out-of-memory (OOM) errors

If you got this error while running a script

```bash
OutOfMemoryError: CUDA out of memory. Tried to allocate 2.22 GiB. GPU 0 has a total capacity of 79.15 GiB of which 228.38 MiB is free. Including non-PyTorch memory, this process
has 78.93 GiB memory in use. Of the allocated memory 76.28 GiB is allocated by PyTorch, and 2.14 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory
is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

it means that your GPU memory size wasn't big enough for the model and script configuration.

Here's a few things you can try:

### Reduce the micro batch size

Adjust the `micro_batch_size = ...` variable in the fine-tuning and pretraining scripts. This variable determines the number of samples loaded per iteration.

A smaller value will simply load fewer samples simultaneously. The minimum value is 1.

Experiment with different micro batch sizes to find a balance between memory consumption and computational efficiency. Smaller micro batch sizes consume less memory but may result in slower training convergence. Conversely, larger micro batch sizes require more memory but can accelerate training speed.

### Reduce the model's context length

The context length (`block_size` in the code) plays a significant role in running models with attention.

* The pretraining scripts are configured to use the full context length of the model to train.
* The finetuning scripts are configured to use the longest sample length of the training data to avoid allocating unnecessary memory (`max_seq_length` in the code).
  If that's longer than the model's context length, an error is raised. If you try to run a batch that is longer than this, an error is raised.

However, your hardware may not support such large context lengths. Here's what you can do:

* For the pretraining scripts, you can simply reduce the `Config(block_size=...)` value.
* For the finetuning scripts, you can trim the length of the samples in your dataset.
  Most of the `scripts/prepare_*.py` scripts expose a `--max_seq_length=...` argument. This might also be useful in cases where
  sample lengths are highly unbalanced, as the presence of a single very long sample would incur a larger memory usage for all other
  shorter samples. For example, the median length of the samples in Alpaca is 110 tokens. Truncating the Alpaca dataset to 256 max tokens reduces the memory requirements of a Falcon 7B model from 23.52 GB to 15.73 GB. For more information about the dataset truncation, please see the *Truncating datasets* section in the the [prepare_datasets.md](prepare_datasets.md) tutorial.

Keep in mind that reducing the context length will affect the modelling performance on text sequences longer than the limit.

### Use lower precision

Our scripts expose the `--precision` argument, this directly impacts the memory usage.

Using true lower precision (`16-true`, `bf16-true`) reduces the memory usage by half compared to `32-true`, however,
the model might start producing NaNs due to the limited range of representable values.

Mixed precision training (`16-mixed`, `bf16-mixed`) provides better stability but offers limited memory reduction.

### Do sharding across multiple GPUs

For exceptionally large models, the aforementioned techniques might still not suffice. If you have multiple GPUs available,
you can trade off memory for speed by changing the `devices = 1` argument in the scripts. Enabling this option enables a parallelism technique (FSDP), sharding the memory across different GPUs.

The default configuration already uses activation checkpointing, but you can enable CPU offloading by changing the `cpu_offload=False` argument in the scripts.

### Try a different optimizer

Our scripts use the [`AdamW` optimizer](https://pytorch.org/docs/main/generated/torch.optim.AdamW.html).
It maintains 2 states for each trainable parameter of the model, meaning that the optimizer memory is double compared to
an optimizer like [`SGD`](https://pytorch.org/docs/main/generated/torch.optim.SGD.html).

You can try replacing it with your optimizer of choice that is lighter in memory requirements. Keep in mind that different optimizers have distinct optimization behaviors, so it's essential to assess their impact on the training process and model performance.
An example would be the recently published [Sophia](https://arxiv.org/abs/2305.14342) or [Lion](https://arxiv.org/abs/2302.06675) optimizers.

This suggestion is particularly relevant for pretraining, as the trainable parameters in the model represent a small
subset of the total in the fine-tuning scripts.
