# LLM Evaluation

## Using lm-evaluation-harness

You can evaluate Lit-GPT using [EleutherAI's lm-eval](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) framework with a large number of different evaluation tasks.

You need to install the `lm-eval` framework first:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

### To evaluate Lit-GPT base models:

```bash
python eval/lm_eval_harness.py \
        --checkpoint_dir "checkpoints/Llama-2-7b-hf/" \
        --precision "bf16-true" \
        --eval_tasks "[truthfulqa_mc,hellaswag]" \
        --batch_size 4 \
        --save_filepath "results.json"
```

## Using HELM

> ![NOTE]\
> acknowledgements to NeurIPS Challenge Organizers and HELM authora for the instructions shown below

### Installing HELM

> ![WARNING]\
> HELM requires Python 3.8\
> It is recommended to install HELM into a virtual environment with Python version 3.8 to avoid dependency conflicts

To create, a Python virtual environment with Python version >= 3.8 and activate it, follow the instructions below.

Install HELM with conda or miniconda, do:

```sh
conda create -n crfm-helm python=3.8 pip -y
conda activate crfm-helm
pip install crfm-helm
```

### Configure HELM

You can configure which datasets to run HELM (Holistic Evaluation of Language Models) on by editing a `run_specs.conf`.

Here's how you can create a simple `run_spec.conf` for local testing:

```sh
echo 'entries: [{description: "mmlu:model=neurips/local,subject=college_computer_science", priority: 4}]' > run_specs.conf
```

> ![NOTE]\
>
> To run your model on a large set of datasets, take a look at the [official example](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/presentation/run_specs_lite.conf) for inspiration.

### Run and Analyze Your Results

After creating `run_spec.conf`, you can run a quick local test with:

```sh
helm-run --conf-paths run_specs.conf --suite v1 --max-eval-instances 10
```

After running the above command, HELM will create a directory named `benchmark_output`. This directory will contain several subdirectories, which are listed below:

- `runs/`
  - `eval_cache/`
  - `mmlu:{SUBJECT}, {METHOD}, {MODEL}/`
- `scenario_instances/`
- `scenarios/`
  - `mmlu`

and then analyze the results with:

```sh
helm-summarize --suite v1
helm-server
```

This will analyze results and then launch a server on your local host, if you're working on a remote machine you might need to setup port forwarding. If everything worked correctly you should see a page that looks like [this](https://user-images.githubusercontent.com/3282513/249620854-080f4d77-c5fd-4ea4-afa4-cf6a9dceb8c9.png)
