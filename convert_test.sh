echo Converting Falcon
python scripts/convert_lit_checkpoint.py --checkpoint_dir=checkpoints/tiiuae/falcon-7b --checkpoint_name=lit_model.pth
echo converting LLaMA
python scripts/convert_lit_checkpoint.py --checkpoint_dir=checkpoints/openlm-research/open_llama_3b --checkpoint_name=lit_model.pth
echo Converting Pythia
python scripts/convert_lit_checkpoint.py --checkpoint_dir=checkpoints/EleutherAI/pythia-1b --checkpoint_name=lit_model.pth
echo Testing
pytest tests/test_convert_lit_checkpoint.py