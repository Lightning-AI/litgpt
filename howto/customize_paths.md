## Customize paths

The project is setup to use specific paths to read the original weights and save checkpoints etc.

For all scripts, you can run

```shell
python script.py -h
```

to get a list of available options. For instance, here's how you would modify the checkpoint dir:

```shell
python scripts/convert_hf_checkpoint.py --ckpt_dir "data/checkpoints/foo"
```

Note that this change will need to be passed along to subsequent steps, for example:

```shell
python scripts/generate.py \
  --checkpoint_path "data/checkpoints/foo/3B/lit-stablelm.pth" \
  --tokenizer_path "data/checkpoints/foo/tokenizer.json"
```

To avoid this, you can use symbolic links to create shortcuts and avoid passing different paths.
