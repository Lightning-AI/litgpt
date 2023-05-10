# Convert HuggingFace (HF) Repository to Parrot

convert_hf_checkpoint parameters:
1. checkpoint_dir: Path = (default is"checkpoints/stabilityai/stablelm-base-alpha-3b"),
2. model_name: Optional[str] = None (Name is gathered from the checkout directory)
3. type: str = "float32" (The default type for HF models)

```bash
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/EleutherAI/pythia-1b
```