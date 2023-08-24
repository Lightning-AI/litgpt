# Preparing Datasets

Currently supported datasets:


| Name | Task | Size | Reference Repo | Paper / Blog | |
|--------------|-------------|---------------------|-----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|---|
| Alpaca | Finetuning | 51,759 samples | [URL](https://github.com/tatsu-lab/stanford_alpaca) | [URL](https://crfm.stanford.edu/2023/03/13/alpaca.html) | |
| Alpaca Libre | Finetuning | 55,370 samples | [URL](https://github.com/mobarski/alpaca-libre) | - | |
| Dolly | Finetuning | 15,011 samples | [URL](https://github.com/databrickslabs/dolly/tree/master/data) | [URL](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) | |
| OpenWeb Text | Pretraining | 8,013,769 documents | [URL](https://github.com/jcpeterson/openwebtext) | [URL](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | |
| RedPajama | Pretraining | 1.2 T tokens | [URL](https://github.com/togethercomputer/RedPajama-Data) | [URL](https://together.ai/blog/redpajama-models-v1) | |

&nbsp;

## Preparing Finetuning Datasets

Note that the dataset needs to be prepared separately for each type of model since the tokenizers used by the models may differ, resulting in slightly different preprocessed datasets.

For the following examples, we will use a Falcon 7B model. However, the same methods are compatible with all other models as well.

The steps here only need to be done once before preparing the finetuning datasets in the following subsections: 

1. Follow the instructions in the [README](../README.md) to install the dependencies.
2. Download and convert the weights following our [guide](download_falcon.md).


&nbsp;

### Alpaca and Alpaca Libre

**Alpaca**

The Alpaca dataset consists of 52,000 instructions and demonstrations produced by OpenAI's text-davinci-003 engine. This data is used in instruction-tuning, helping improve the performance of language models to follow instructions.

In its development, the creators leveraged the data generation methodology from the [Self-Instruct framework](https://github.com/yizhongw/self-instruct).

The original [Alpaca] dataset can be prepared as follows:

```bash
python scripts/prepare_alpaca.py \
 --checkpoint_dir checkpoints/tiiuae/falcon-7b
```

**Alpaca Libre**

[Alpaca Libre](https://github.com/mobarski/alpaca-libre) is a reimplementation or alternative to Alpaca using the same formatting.

To use Alpaca Libre instead of the original Alpaca dataset, use the following command:


```bash
python scripts/prepare_alpaca.py \
 --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
 --data_file_url "https://raw.githubusercontent.com/mobarski/alpaca-libre/main/data/output/alpaca_libre_ok_tasks_v4.json" \
 --data_file_name "alpaca_libre_data_cleaned_archive.json" \
 --destination_path "data/alpaca_libre"
```

**Finetuning**

After preparing the dataset, you can finetune the model using the [`finetune/*.py`](../finetune/) scripts, for example,

```bash
python finetune/lora.py
 --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
 --data_dir "data/alpaca_libre" \
 --out_dir "out/lora/alpaca"
```

**It's important to make sure that the `prepare_*.py` and `finetune/*.py` scripts use the same model via `--checkpoint_dir`**

Please read the [tutorials/finetune_*.md](../tutorials) documents for more information about finetuning models.


&nbsp;

### Dolly

The Dolly dataset is a publicly available collection of 15k instruction-following entries created by Databricks. It spans multiple behavioral domains, as described in the [InstructGPT paper](https://arxiv.org/abs/2203.02155). These include areas like brainstorming, classification, closed QA, content creation, information retrieval, open QA, and summary generation.

The usage is similar to the Alpaca dataset described above. Using Falcon 7b as an example, we can prepare the dataset as follows:

```bash
python scripts/prepare_dolly.py \
 --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
 --data_file_url "https://raw.githubusercontent.com/mobarski/alpaca-libre/main/data/output/alpaca_libre_ok_tasks_v4.json" \
 --data_file_name "alpaca_libre_data_cleaned_archive.json" \
 --destination_path "data/dolly"
```


## Preparing Pretraining Datasets

In addition to the finetuning dataset described above, Lit-GPT also supports several datasets for pretraining. The pretraining datasets are described in more detail in the following separate tutorial documents:

- [Pretrain Llama 2 on OpenWebText](./pretrain_openwebtext.md)
- [Pretrain Llama 2 on RedPajama](./pretrain_redpajama.md)

