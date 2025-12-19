from litgpt.data.indic_instruct import IndicInstruct
from litgpt.tokenizer import Tokenizer

tokenizer = Tokenizer("checkpoints/soketlabs/pragna-1b")

indic_instruct = IndicInstruct()

indic_instruct.connect(tokenizer, max_seq_length=2048)
indic_instruct.prepare_data()
indic_instruct.setup()

# train_dataloader = indic_instruct.train_dataloader()
train_dataset = indic_instruct.train_dataset

print(train_dataset[0])

