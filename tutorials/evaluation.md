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

### To evaluate LoRA finetuned LLMs:

```bash
python eval/lm_eval_harness_lora.py \
        --lora_path "lit_model_lora_finetuned.pth" \
        --checkpoint_dir "checkpoints/Llama-2-7b-hf/" \
        --precision "bf16-true" \
        --eval_tasks "[truthfulqa_mc,hellaswag]" \
        --batch_size 4 \
        --save_filepath "results.json"
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
