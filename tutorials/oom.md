## Dealing with out-of-memory (OOM) errors

If you got this error while running a script

```bash
OutOfMemoryError: CUDA out of memory. Tried to allocate 2.22 GiB. GPU 0 has a total capacty of 79.15 GiB of which 228.38 MiB is free. Including non-PyTorch memory, this process
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

The context length plays a significant role in running models with attention. By default, the scripts use the maximum
context length of the model, or a shorter length if the data is smaller. However, your hardware may not support such large context lengths.

To manually reduce it, you can modify the `max_seq_length` argument passed to the model forward.
Particularly, for the fine-tuning scripts, you can modify the `override_max_seq_length = None` at the beginning of the script.

Keep in mind that reducing the context length will affect the model's learning ability by limiting the attention window.

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
