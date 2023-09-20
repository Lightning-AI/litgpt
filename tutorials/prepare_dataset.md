# Preparing Datasets

Below is a table of all datasets that are currently supported in Lit-GPT:


| Name         | Task        | Size                | Reference Repo                                                  | Paper / Blog                                                                                                              | Data License                                                                                                                                                                                                     |
|--------------|-------------|---------------------|-----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Alpaca       | Finetuning  | 51,759 samples      | [URL](https://github.com/tatsu-lab/stanford_alpaca)             | [URL](https://crfm.stanford.edu/2023/03/13/alpaca.html)                                                                   | Attribution-NonCommercial 4.0 International, [ URL](https://crfm.stanford.edu/2023/03/13/alpaca.html)                                                                                                            |
| Alpaca Libre | Finetuning  | 55,370 samples      | [URL](https://github.com/mobarski/alpaca-libre)                 | -                                                                                                                         | CC0/MIT,  [URL](https://github.com/mobarski/alpaca-libre)                                                                                                                                                        |
| Dolly        | Finetuning  | 15,011 samples      | [URL](https://github.com/databrickslabs/dolly/tree/master/data) | [URL](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)              | CC-BY-SA, [URL](https://github.com/databrickslabs/dolly#model-overview)                                                                                                                                          |
| LongForm     | Finetuning  | 23,652 samples      | [URL](https://github.com/akoksal/LongForm)                      | [URL](https://arxiv.org/abs/2304.08460)                                                                                   | No information provided and subset-dependent, [URL](https://github.com/akoksal/LongForm) |
| LIMA         | Finetuning  | 1,084 samples       | [URL](https://huggingface.co/datasets/GAIR/lima)                | [URL](https://arxiv.org/abs/2305.11206)                                                                                   | "If the source data of LIMA has a stricter license than CC BY-NC-SA, the LIMA dataset follows the same. Otherwise, it follows the CC BY-NC-SA license", [URL](https://huggingface.co/datasets/GAIR/lima#license) |
| OpenWeb Text | Pretraining | 8,013,769 documents | [URL](https://github.com/jcpeterson/openwebtext)                | [URL](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Unspecified                                                                                                                                                                                                      |
| RedPajama    | Pretraining | 1.2 T tokens        | [URL](https://github.com/togethercomputer/RedPajama-Data)       | [URL](https://together.ai/blog/redpajama-models-v1)                                                                       | Subset-dependent, [URL](https://github.com/togethercomputer/RedPajama-Data#license)                                                                                                                              |                                                                     |   |

&nbsp;

## Preparing Finetuning Datasets

Note that the dataset needs to be prepared separately for each type of model since the tokenizers used by the models may differ, resulting in slightly different preprocessed datasets.

For the following examples, we will use a Falcon 7B model. However, the same methods are compatible with all other models as well.

The steps here only need to be done once before preparing the finetuning datasets in the following subsections:

1. Follow the instructions in the [README](../README.md) to install the dependencies.
2. Download and convert the weights following our [guide](download_falcon.md).

&nbsp;

### Alpaca and Alpaca Libre

&nbsp;

**Alpaca**

The Alpaca dataset consists of 52,000 instructions and demonstrations produced by OpenAI's text-davinci-003 engine. This data is used in instruction-tuning, helping improve the performance of language models to follow instructions.

In its development, the creators leveraged the data generation methodology from the [Self-Instruct framework](https://github.com/yizhongw/self-instruct).

The original [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) dataset can be prepared as follows:

```bash
python scripts/prepare_alpaca.py \
 --checkpoint_dir checkpoints/tiiuae/falcon-7b
```

&nbsp;

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

&nbsp;

### Dolly

The Dolly dataset is a publicly available collection of 15k instruction-following entries created by Databricks. It spans multiple behavioral domains, as described in the [InstructGPT paper](https://arxiv.org/abs/2203.02155) paper. These include areas like brainstorming, classification, closed QA, content creation, information retrieval, open QA, and summary generation.

The usage is similar to the Alpaca dataset described above. Using Falcon 7b as an example, we can prepare the dataset as follows:

```bash
python scripts/prepare_dolly.py \
 --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
```

&nbsp;

### LIMA

The LIMA dataset is a collection of 1,000 carefully curated prompts and responses, as described in the [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) paper. The dataset is sourced from three community Q&A websites: Stack Exchange, wikiHow, and the Pushshift Reddit Dataset. In addition, it also contains prompts and answers written and collected by the authors of the LIMA paper.

The usage is similar to the Dolly dataset described above except that it requires an Hugging Face access token that you need to copy & paste from your Hugging Face account. Using Falcon 7b as an example, we can prepare the dataset as follows:

```bash
python scripts/prepare_lima.py \
 --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
 --access_token "insert_your_token_here"
```

LIMA contains a handful of multiturn conversations. By default, only the first instruction-response pairs from
each of these multiturn conversations are included. If you want to override this behavior and include the follow up instructions
and responses, set `--include_multiturn_conversations True`.


&nbsp;

### LongForm

LongForm is a semi-synthetic dataset based on raw text corpora for which the instructions were generated via an LLM. For more details about the instruction-generation process, please refer to the [LongForm research paper](https://arxiv.org/abs/2304.08460) by Köksal et al. According to the research paper, a Llama 7B model trained on LongForm achieves substantially better performance than the same Llama model trained on the 2x larger Alpaca dataset.

LongForm consists of 23,652 training samples, 2,042 validation samples, and 2,045 test samples. (In Lit-GPT, the validation samples are currently not used.)

The more detailed dataset composition is as follows based on a table taken from the [dataset repository](https://github.com/akoksal/LongForm):

| **Type**               | **Source**     | **Number of Examples** |
|------------------------|----------------|------------------------|
| **Corpora**            | C4             | 10,000                 |
|                        | Wikipedia      | 5,000                  |
| **Structured Corpora** | Stack Exchange | 4,380                  |
|                        | WikiHow        | 2,500                  |
| **Tasks**              | NIv2           | 3,684                  |
|                        | Big Bench      | 600                    |
|                        | BEA-GEC        | 1,203                  |
|                        | Enron          | 372                    |
| **Total**              |                | 27,739                 |
|  |   |  |
| **Train**              |                | 23,652                 |
| **Validation**         |                | 2,042                  |
| **Test**               |                | 2,045                  |

License information is not provided but would depend on the individual subsets listed above.


&nbsp;

## Finetuning After Data Preparation

After preparing the dataset, you can finetune the model using the [`finetune/*.py`](../finetune/) scripts, for example,

```bash
python finetune/lora.py
 --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
 --data_dir "data/alpaca_libre" \
 --out_dir "out/lora/alpaca"
```

Please read the [tutorials/finetune_*.md](../tutorials) documents for more information about finetuning models.

> [!IMPORTANT]
> Make sure that the `prepare_*.py` and `finetune/*.py` scripts use the same model checkpoint specified via `--checkpoint_dir`.

> [!IMPORTANT]
> By default, the maximum sequence length is obtained from the model configuration file. In case you run into out-of-memory errors, especially in the cases of LIMA and Dolly,
> you can try to lower the context length by editing the  [`finetune/lora.py` file](https://github.com/Lightning-AI/lit-gpt/blob/main/finetune/lora.py#L37) and change `override_max_seq_length = None` to `override_max_seq_length = 2048`.

&nbsp;

## Preparing Pretraining Datasets

In addition to the finetuning dataset described above, Lit-GPT also supports several datasets for pretraining. The pretraining datasets are described in more detail in the following separate tutorial documents:

- [Pretrain Llama 2 on OpenWebText](./pretrain_openwebtext.md)
- [Pretrain Llama 2 on RedPajama](./pretrain_redpajama.md)
