# LLM Evaluation

## Using lm-evaluation-harness

You can evaluate Lit-GPT using [EleutherAI's lm-eval](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) framework with a large number of different evaluation tasks.

You need to install the `lm-eval` framework first:

```bash
pip install https://github.com/EleutherAI/lm-evaluation-harness/archive/refs/heads/master.zip -U
```

### Evaluating Lit-GPT base models

Use the following command to evaluate Lit-GPT models on all tasks in Eleuther AI's Evaluation Harness.

```bash
python eval/lm_eval_harness.py \
        --checkpoint_dir "checkpoints/Llama-2-7b-hf/" \
        --precision "bf16-true" \
        --batch_size 4 \
        --save_filepath "results.json"
```

To evaluate on LLMs on specific tasks, for example, TruthfulQA and HellaSwag, you can use the `--eval_task` flag as follows:

```bash
python eval/lm_eval_harness.py \
        --checkpoint_dir "checkpoints/Llama-2-7b-hf/" \
        --eval_tasks "[truthfulqa_mc,hellaswag]" \
        --precision "bf16-true" \
        --batch_size 4 \
        --save_filepath "results.json"
```

A list of supported tasks can be found [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md).



### Evaluating LoRA-finetuned LLMs

The above command can be used to evaluate models that are saved via a single checkpoint file. This includes downloaded checkpoints and base models finetuned via the full and adapter finetuning scripts. For LoRA-finetuned models, use the `lm_eval_harness_lora.py` script instead:

```bash
python eval/lm_eval_harness_lora.py \
        --lora_path "lit_model_lora_finetuned.pth" \
        --checkpoint_dir "checkpoints/Llama-2-7b-hf/" \
        --precision "bf16-true" \
        --eval_tasks "[truthfulqa_mc,hellaswag]" \
        --batch_size 4 \
        --save_filepath "results.json"
```

## Using Holistic Evaluation of Language Models (HELM)

You can evaluate Lit-GPT using [HELM](https://crfm.stanford.edu/helm/latest/), a benchmark that aims to improve the transparency of language models.

You need to install the `helm` framework first:

```bash
pip install git+https://github.com/stanford-crfm/helm.git@main
```

**Step 1:** Run Lit-GPT as API

```shell
python eval/helm/main.py \
        --checkpoint_dir "checkpoints/Llama-2-7b-hf/" \
        --precision "bf16-true"
```

**Step 2:** Evaluate with HELM CLI

Create a `run_specs.conf` file that contains tasks on which you want to evaluate your LLM. You can find a list of tasks [here](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/presentation/run_specs.conf).

```shell
echo 'entries: [{description: "mmlu:model=neurips/local,subject=college_computer_science", priority: 4}]' > run_specs.conf
helm-run --conf-paths run_specs.conf --suite v1 --max-eval-instances 1000
helm-summarize --suite v1
```


## FAQs

* **How do I evaluate on MMLU?**

  MMLU is available as with lm-eval harness but the task name is not MMLU. You can use `hendrycksTest*` as regex to evaluate on MMLU.
  ```shell
  python eval/lm_eval_harness_lora.py \
          --lora_path "lit_model_lora_finetuned.pth" \
          --checkpoint_dir "checkpoints/Llama-2-7b-hf/" \
          --precision "bf16-true" \
          --eval_tasks "[hendrycksTest*]" \
          --batch_size 4 \
          --num_fewshot 5 \
          --save_filepath "results.json"
  ```


* **Is Truthful MC is not available in lm-eval?**

  It is available as `truthfulqa_mc`.
