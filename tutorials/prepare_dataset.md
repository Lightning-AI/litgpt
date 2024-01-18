# Preparing Datasets

Below is a table of all datasets that are currently supported in Lit-GPT:

| Name         | Task        | Size                | Reference Repo                                                  | Paper / Blog                                                                                                              | Data License                                                                                                                                                                                                     |
|--------------|-------------|---------------------|-----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Alpaca       | Finetuning  | 51,759 samples      | [URL](https://github.com/tatsu-lab/stanford_alpaca)             | [URL](https://crfm.stanford.edu/2023/03/13/alpaca.html)                                                                   | Attribution-NonCommercial 4.0 International, [URL](https://crfm.stanford.edu/2023/03/13/alpaca.html)                                                                                                             |
| Alpaca Libre | Finetuning  | 55,370 samples      | [URL](https://github.com/mobarski/alpaca-libre)                 | -                                                                                                                         | CC0/MIT,  [URL](https://github.com/mobarski/alpaca-libre)                                                                                                                                                        |
| Dolly        | Finetuning  | 15,011 samples      | [URL](https://github.com/databrickslabs/dolly/tree/master/data) | [URL](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)              | CC-BY-SA, [URL](https://github.com/databrickslabs/dolly#model-overview)                                                                                                                                          |
| FLAN         | Finetuning  | 1,753,240 samples   | [UR](https://huggingface.co/datasets/Muennighoff/flan)          | [URL](https://blog.research.google/2023/02/the-flan-collection-advancing-open.html)                                       | Subset dependent                                                                                                                                                                                                 |
| LongForm     | Finetuning  | 23,652 samples      | [URL](https://github.com/akoksal/LongForm)                      | [URL](https://arxiv.org/abs/2304.08460)                                                                                   | No information provided and subset-dependent, [URL](https://github.com/akoksal/LongForm)                                                                                                                         |
| LIMA         | Finetuning  | 1,084 samples       | [URL](https://huggingface.co/datasets/GAIR/lima)                | [URL](https://arxiv.org/abs/2305.11206)                                                                                   | "If the source data of LIMA has a stricter license than CC BY-NC-SA, the LIMA dataset follows the same. Otherwise, it follows the CC BY-NC-SA license", [URL](https://huggingface.co/datasets/GAIR/lima#license) |
| OpenWeb Text | Pretraining | 8,013,769 documents | [URL](https://github.com/jcpeterson/openwebtext)                | [URL](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Unspecified                                                                                                                                                                                                      |
| RedPajama    | Pretraining | 1.2 T tokens        | [URL](https://github.com/togethercomputer/RedPajama-Data)       | [URL](https://together.ai/blog/redpajama-models-v1)                                                                       | Subset-dependent, [URL](https://github.com/togethercomputer/RedPajama-Data#license)                                                                                                                              |

&nbsp;

## Preparing Finetuning Datasets

Note that the dataset needs to be prepared separately for each type of model since the tokenizers used by the models may differ, resulting in slightly different preprocessed datasets.

For the following examples, we will use a Falcon 7B model. However, the same methods are compatible with all other models as well.

The steps here only need to be done once before preparing the finetuning datasets in the following subsections:

1. Follow the instructions in the [README](../README.md) to install the dependencies.
2. Download and convert the weights following our [guide](download_falcon.md).

&nbsp;

### Alpaca

&nbsp;

The Alpaca dataset consists of 52,000 instructions and demonstrations produced by OpenAI's text-davinci-003 engine. This data is used in instruction-tuning, helping improve the performance of language models to follow instructions.

In its development, the creators leveraged the data generation methodology from the [Self-Instruct framework](https://github.com/yizhongw/self-instruct).

The original [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) dataset can be prepared as follows:

```bash
python scripts/prepare_alpaca.py \
 --checkpoint_dir checkpoints/tiiuae/falcon-7b
```

#### Truncating datasets

By default, the finetuning script (for example [`finetuning/lora.py`](../finetuning/lora.py)) will determine the size of the longest tokenized sample in the dataset to determine the block size. However, if you are willing to truncate a few examples in the training set, you can reduce the computational resource requirements significantly. For instance you can set a sequence length threshold via `--max_seq_length`. We can determine an appropriate maximum sequence length by considering the distribution of the data sample lengths shown in the histogram below.

<img src="images/prepare_dataset/alpaca.jpg" width=400px>

In this case, a cut-off of 256 may be a reasonable choice:

```bash
python scripts/prepare_alpaca.py \
 --checkpoint_dir checkpoints/tiiuae/falcon-7b \
 --max_seq_length 256
```

For comparison, the Falcon 7B model requires 23.52 GB of memory for the original Alpaca dataset and 15.73 GB of memory for the truncated Alpaca dataset when finetuning with LoRA using a micro batchsize of 1 and bfloat-16 precision.

&nbsp;

### Alpaca Libre

[Alpaca Libre](https://github.com/mobarski/alpaca-libre) is a reimplementation or alternative to Alpaca using the same formatting.

To use Alpaca Libre instead of the original Alpaca dataset, use the following command:

```bash
python scripts/prepare_alpaca.py \
 --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
 --data_file_url "https://raw.githubusercontent.com/mobarski/alpaca-libre/main/data/output/alpaca_libre_ok_tasks_v4.json" \
 --data_file_name "alpaca_libre_data_cleaned_archive.json" \
 --destination_path "data/alpaca_libre"
```

The Alpaca Libre dataset distribution is shown below.

<img src="images/prepare_dataset/alpaca_libre.jpg" width=400px>

You may want to consider truncating the dataset (see the *Truncating datasets* discussion in the Alpaca section for more information.) For this dataset, a cut-off of 256 may be a good choice:

```bash
python scripts/prepare_alpaca.py \
 --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
 --data_file_url "https://raw.githubusercontent.com/mobarski/alpaca-libre/main/data/output/alpaca_libre_ok_tasks_v4.json" \
 --data_file_name "alpaca_libre_data_cleaned_archive.json" \
 --destination_path "data/alpaca_libre" \
 --max_seq_length 256
```

&nbsp;

### Dolly

The Dolly dataset is a publicly available collection of 15k instruction-following entries created by Databricks. It spans multiple behavioral domains, as described in the [InstructGPT paper](https://arxiv.org/abs/2203.02155) paper. These include areas like brainstorming, classification, closed QA, content creation, information retrieval, open QA, and summary generation.

The usage is similar to the Alpaca dataset described above. Using Falcon 7b as an example, we can prepare the dataset as follows:

```bash
python scripts/prepare_dolly.py \
 --checkpoint_dir "checkpoints/tiiuae/falcon-7b"
```

The Dolly dataset distribution is shown below.

<img src="images/prepare_dataset/dolly.jpg" width=400px>

You may want to consider truncating the dataset (see the *Truncating datasets* discussion in the Alpaca section for more information.) For this dataset, a cut-off of 512 may be a good choice:

```bash
python scripts/prepare_dolly.py \
 --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
 --max_seq_length 512
```

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
|                        |                |                        |
| **Train**              |                | 23,652                 |
| **Validation**         |                | 2,042                  |
| **Test**               |                | 2,045                  |

License information is not provided but would depend on the individual subsets listed above.

The LongForm dataset distribution is shown below.

<img src="images/prepare_dataset/longform.jpg" width=400px>

You may want to consider truncating the dataset (see the *Truncating datasets* discussion in the Alpaca section for more information.) For this dataset, a cut-off of 1500 may be a good choice:

```bash
python scripts/prepare_dolly.py \
 --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
 --max_seq_length 1500
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

The Lima dataset distribution is shown below.

<img src="images/prepare_dataset/lima.jpg" width=400px>

You may want to consider truncating the dataset (see the *Truncating datasets* discussion in the Alpaca section for more information.) For this dataset, a cut-off of 512 may be a good choice:

```bash
python scripts/prepare_dolly.py \
 --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
 --max_seq_length 512
```

&nbsp;

### FLAN

FLAN is a collection of several datset subsets by Google. In particular, the provided script in Lit-GPT loads the subsets from
[here](https://huggingface.co/datasets/Muennighoff/flan).

By default, all subsets (1,386,050 samples) and validations sets (367,190 subsets) are combined into a single dataset:

```bash
python scripts/prepare_flan.py \
  --checkpoint_dir "checkpoints/tiiuae/falcon-7b"
```

However, you can also select individual subsets via comma-separated strings as follows:


```bash
python scripts/prepare_flan.py \
  --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
  --subsets "aeslc_10templates,ag_news_subset_10templates,anli_r1_10templates"
```

You can find a list of all 66 supported subsets [here](https://huggingface.co/datasets/Muennighoff/flan).

&nbsp;

### Finetuning After Data Preparation

After preparing the dataset, you can finetune the model using the [`finetune/*.py`](../finetune/) scripts, for example,

```bash
python finetune/lora.py
 --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
 --data_dir "data/alpaca_libre" \
 --out_dir "out/lora/alpaca"
```

Please read the [tutorials/finetune_*.md](.) documents for more information about finetuning models.

> [!IMPORTANT]
> Make sure that the `prepare_*.py` and `finetune/*.py` scripts use the same model checkpoint specified via `--checkpoint_dir`.

> [!IMPORTANT]
> By default, the maximum sequence length is obtained from the model configuration file. In case you run into out-of-memory errors, especially in the cases of LIMA and Dolly,
> you can try to lower the context length by preparing the dataset with a fixed max length, for example, `python scripts/prepare_lima.py --max_seq_length 2048`. For more information on truncating datasets, see the *Truncating datasets* section in the Alpaca section near the top of this article.

&nbsp;

## Preparing Custom Datasets for Instruction Finetuning

The models in Lit-GPT expect datasets for instruction finetuning in the following format:

```text
[
    {
        "instruction": "Write a limerick about a
                        pelican.”,
        "input": "",
        "output": "There once was a pelican so fine,
                   \nHis beak was as colorful as
                   sunshine,\nHe would fish all day,\nIn
                   a very unique way,\nThis pelican was
                   truly divine!\n\n\n"
    },
    {
        "instruction": "Identify the odd one out from
                        the group.",
        "input": "Carrot, Apple, Banana, Grape",
        "output": "Carrot\n\n"
    },
]
```

(Note that depending on the task, the `"input"` text can be an empty string, as shown above.)

Custom datasets can be prepared by either creating a new `scripts/prepare_dataset.py` script or reading the dataset
from a CSV file.

&nbsp;

### Preparing Custom Datasets From a CSV File

You can prepare custom dataset using a CSV file with the following columns:

- `instruction`: Column which will describe the task.
- `input`: A string holding a special input value for the instruction. This applies to some samples, and in others, this is empty (empty string).
- `output`: The expected response

> If any of the columns is missing, then the script will fail to create the dataset.

Before you finetune, prepare the dataset using the `prepare_csv.py` script:

```bash
python scripts/prepare_csv.py --csv_path path/to/the/file.csv
```

You can also customize the dataset generation by using these additional parameters

- `destination_path`: The folder where the binary data will be saved. By default, it is saved inside `data/csv`

- `checkpoint_dir`: The model checkpoint dir. It will use the model's tokenizer to load and convert the string to input ids. Defaults to `"checkpoints/stabilityai/stablelm-base-alpha-3b"`

- `test_split_fraction`: The fraction of the data to split. Defaults to `0.1`

- `seed`: The seed value to reproduce the same random splits for train and test data.

- `mask_inputs`: Whether we require any masking or not.

- `ignore_index`: Mask out all the tokens after this index when preparing the dataset.

To use the the settings described above, you can add the respective command line arguments when calling `prepare_csv.py` as shown in the example below:

```bash
python scripts/prepare_csv.py --csv_path test_data.csv \
--destination_path data/csv \
--checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b \
--test_split_fraction 0.1 \
--seed 42 \
--mask_inputs false \
--ignore_index -1
```

Replace `test_data.csv` with your CSV path and the other additional parameters accordingly. Executing the command above will save `train.pt` and `test.pt` on your disk at the `destination_path`. Now you can use the prepared data to [finetune your model](finetune_lora.md#running-the-finetuning).

&nbsp;

### Preparing Custom Datasets Using a Dataset Preparation Script

If you don't have a CSV file following the format described in the previous section, the easiest way to prepare a new dataset is to copy and modify one of the existing dataset preparation scripts:

- [`scripts/prepare_alpaca.py`](https://github.com/Lightning-AI/lit-gpt/blob/main/scripts/prepare_alpaca.py) (if you plan to load a dataset from a JSON file);
- [`scripts/prepare_lima.py`](https://github.com/Lightning-AI/lit-gpt/blob/main/scripts/prepare_lima.py) (if you plan to load a dataset using the `datasets` Python library).

These scripts may look intimidating at first glance since they include code for tokenizing the dataset for a specific LLM that is provided via a checkpoint directory. However, note that you only need to modify a small fraction of the code file, namely the portion that downloads and formats the training data.

In [`scripts/prepare_lima.py`](https://github.com/Lightning-AI/lit-gpt/blob/main/scripts/prepare_lima.py), the [line 26](https://github.com/Lightning-AI/lit-gpt/blob/98fad263a62e5e57821de817bdd5e316abfb34d4/scripts/prepare_lima.py#L26) references the HF repo ID, and the lines [50-53](https://github.com/Lightning-AI/lit-gpt/blob/98fad263a62e5e57821de817bdd5e316abfb34d4/scripts/prepare_lima.py#L50-L53) save the dataset as `train_data`. Here, `train_data` is a list that contains the instruction examples in the format mentioned above.

In [`scripts/prepare_alpaca.py`](https://github.com/Lightning-AI/lit-gpt/blob/main/scripts/prepare_alpaca.py), you only need to modify [lines 24-25](https://github.com/Lightning-AI/lit-gpt/blob/98fad263a62e5e57821de817bdd5e316abfb34d4/scripts/prepare_alpaca.py#L24-L25) for the file name and URL, assuming the JSON file you are working with has the same format as the [Alpaca JSON file](https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json).

&nbsp;

## Preparing Pretraining Datasets

In addition to the finetuning dataset described above, Lit-GPT also supports several datasets for pretraining. The pretraining datasets are described in more detail in the following separate tutorial documents:

- [Pretrain Llama 2 on OpenWebText](./pretrain_openwebtext.md)
- [Pretrain Llama 2 on RedPajama](./pretrain_redpajama.md)
