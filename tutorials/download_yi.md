## Download Yi weights

The Yi series models are the next generation of open-source large language models trained from scratch by 01.AI. For more details, see the official [README](https://github.com/01-ai/Yi).

To see all available versions, run:

```bash
python scripts/download.py | grep Yi
```

which will print

```text
01-ai/Yi-6B-Chat
01-ai/Yi-34B-Chat
```

Download the weights and convert the checkpoint to the lit-gpt format (eg. 01-ai/Yi-6B-Chat):

```bash
pip install huggingface_hub

python scripts/download.py --repo_id 01-ai/Yi-6B-Chat --from_safetensors=True

python scripts/convert_hf_checkpoint.py \
    --checkpoint_dir checkpoints/01-ai/Yi-6B-Chat
```

-----

You're done! To execute the model just run:

```bash
pip install sentencepiece

python chat/base.py --checkpoint_dir ./checkpoints/01-ai/Yi-6B-Chat  --precision "bf16-true"
```

Chat example (with chat history):

```bash
>> Prompt: hi
>> Reply: Hello! How can I assist you today?
Time for inference: 0.65 sec total, 13.93 tokens/sec, 9 tokens, prompt length 10

>> Prompt: 你是谁
>> Reply: My name is Yi, and I am a language model based on the transformers architecture developed by 01.AI. My purpose is to be a helpful resource for you, capable of answering questions and offering insightful information across a wide range of topics. How may I help you?
Time for inference: 1.55 sec total, 37.36 tokens/sec, 58 tokens, prompt length 32

>> Prompt: 床前明月光
>> Reply: 床前明月光 (Before the bed shines brightly) 常被理解为唐代诗人李白《静夜思》中的名句，表达了诗人夜晚独处时，看到窗户前洒满月光而产生的思乡之情。这句诗经常被用作中国人思念家乡或亲人时的表达方式。
Time for inference: 1.62 sec total, 37.63 tokens/sec, 61 tokens, prompt length 105
```
