# LLM Evaluation

&nbsp;

## Using lm-evaluation-harness

You can evaluate Lit-GPT using [EleutherAI's lm-eval](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) framework with a large number of different evaluation tasks.

You need to install the `lm-eval` framework first:

```bash
pip install https://github.com/EleutherAI/lm-evaluation-harness/archive/refs/heads/master.zip -U
```

&nbsp;

### Evaluating Lit-GPT base models

Use the following command to evaluate Lit-GPT models on all tasks in Eleuther AI's Evaluation Harness.

```bash
python eval/lm_eval_harness.py \
    --checkpoint_dir "checkpoints/meta-llama/Llama-2-7b-hf" \
    --precision "bf16-true" \
    --save_filepath "results.json"
```

To evaluate on LLMs on specific tasks, for example, TruthfulQA and HellaSwag, you can use the `--eval_task` flag as follows:

```bash
python eval/lm_eval_harness.py \
    --checkpoint_dir "checkpoints/meta-llama/Llama-2-7b-hf" \
    --eval_tasks "[truthfulqa_mc,hellaswag]" \
    --precision "bf16-true" \
    --save_filepath "results.json"
```

A list of supported tasks can be found [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md).

&nbsp;

### Evaluating LoRA-finetuned LLMs

The above command can be used to evaluate models that are saved via a single checkpoint file. This includes downloaded checkpoints and base models finetuned via the full and adapter finetuning scripts.

For LoRA-finetuned models, you need to first merge the LoRA weights with the original checkpoint file as described in the [Merging LoRA Weights](finetune_lora.md#merging-lora-weights) section of the LoRA finetuning documentation.

&nbsp;

## FAQs

* **How do I evaluate on MMLU?**

  MMLU is available as with lm-eval harness but the task name is not MMLU. You can use `hendrycksTest*` as regex to evaluate on MMLU.

  ```shell
  python eval/lm_eval_harness.py \
      --checkpoint_dir "checkpoints/meta-llama/Llama-2-7b-hf" \
      --precision "bf16-true" \
      --eval_tasks "[hendrycksTest*]" \
      --num_fewshot 5 \
      --save_filepath "results.json"
  ```

* **Is Truthful MC is not available in lm-eval?**

  It is available as `truthfulqa_mc`.
