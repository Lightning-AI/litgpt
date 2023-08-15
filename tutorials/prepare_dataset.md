# Preparing Datasets

Currently supported datasets:


| Name         | Task        | Samples | Size         | Reference Repo                                            | Paper                                                                                                                     |
|--------------|-------------|---------|--------------|-----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| Alpaca       | Finetuning  | 51,759  |              |                                                           |                                                                                                                           |
| Alpaca Libre | Finetuning  | 55,370  |              |                                                           |                                                                                                                           |
| OpenWeb Text | Pretraining | -       |              | [URL](https://github.com/jcpeterson/openwebtext)          | [URL](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |
| RedPajama    | Pretraining | -       | 1.2 T tokens | [URL](https://github.com/togethercomputer/RedPajama-Data) |                                                                                                                           |                                                                                                                 |


&nbsp;

## Preparing Finetuning Datasets

Note that the dataset needs to prepared separate for each type of model since the tokenizers used by the models may differ, resulting in slightly differnt preprocessed datasets.

For the following examples, we are going to use a Falcon 7B model. However, the same methods are compatible with all other models as well.

The steps here only need to be done once before preparing the finetuning datasets in the following subsections: 

1. Follow the instructions in the [README](../README.md) to install the dependencies.
2. Download and convert the weights following our [guide](download_falcon.md).


&nbsp;

### Alpaca and Alpaca Libre

The regular Alpaca dataset can be prepared as follows:

```bash
python scripts/prepare_alpaca.py --checkpoint_dir checkpoints/tiiuae/falcon-7b
```

To use Alpaca Libre instead, use the following command:

```bash
python scripts/prepare_alpaca.py --checkpoint_dir checkpoints/tiiuae/falcon-7b \
--data_file_url "https://raw.githubusercontent.com/mobarski/alpaca-libre/main/data/output/alpaca_libre_ok_tasks_v4.json" \
--data_file_name "alpaca_libre_data_cleaned_archive.json" \
--destination_path "data/alpaca_libre"
```

&nbsp;

### Custom Finetuning Datasets

Todo


## Preparing Pretraining Datasets

### OpenWebText

### Red Pajama