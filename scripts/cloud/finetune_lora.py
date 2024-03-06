from lit_gpt.cloud.finetune.lora import FinetuneLora

client = FinetuneLora(api_key="...")
client.run(data_path="data/data.json")