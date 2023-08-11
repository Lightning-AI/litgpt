# LLM Evaluation

## Using lm-evaluation-harness

You can evaluate Lit-GPT using [EleutherAI's lm-eval](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) framework with a large number of different evaluation tasks.

You need to install the `lm-eval` framework first:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
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


```python
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
