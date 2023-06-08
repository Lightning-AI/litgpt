NVIDIA's TransformerEngine support requires a few specific installation steps:

```bash
pip install git+https://github.com/Lightning-AI/lightning.git@carmocca/transformer-engine
# you'll want CUDA 12.1
pip install --index-url https://download.pytorch.org/whl/nightly/cu121 --pre 'torch>=2.1.0dev'
# needs to be installed separately until https://github.com/HazyResearch/flash-attention/issues/246 is resolved
pip install flash-attn --no-build-isolation
NVTE_FRAMEWORK=pytorch pip install git+https://github.com/NVIDIA/TransformerEngine.git@main
```

Baseline speed
TE layers speed (no autocast)
TE + autocast speed
Fabric automatic replacement speed (just linear)