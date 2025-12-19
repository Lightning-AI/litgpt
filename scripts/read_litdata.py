import os
from litdata import StreamingDataset, StreamingDataLoader, TokensLoader
from tqdm import tqdm
from litgpt.tokenizer import Tokenizer

dataset = StreamingDataset(
  input_dir=f"/raid/data/sangraha-processed-data-v2",
#   item_loader=TokensLoader(block_size=2048 + 1),
  shuffle=True,
  drop_last=True,
)

# train_dataloader = StreamingDataLoader(dataset, batch_size=8, pin_memory=True, num_workers=os.cpu_count())

tokenizer = Tokenizer("checkpoints/soketlabs/pragna-1b")

for i in range(1000000, 1000000+10):
    print(dataset[i])
    print(tokenizer.decode(dataset[i]))
    print("--------------------------------")