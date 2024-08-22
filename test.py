import torch
import litgpt
from torch import tensor

from litgpt.generate.base import generate_fn, next_token
from litgpt.api import LLM, GPT

input_pos = tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32, device="cuda:0")
# input_pos = input_pos.reshape(-1, input_pos.size(0))
x = tensor(
    [43993, 25, 1867, 466, 32660, 17485, 4483, 30, 198, 26410],
    device="cuda:0",
    dtype=torch.int32,
)
x = x.reshape(-1, x.size(0))

print(input_pos)
print(x)
print(input_pos.shape)
print(x.shape)

llm: LLM = LLM.load("microsoft/phi-2")
model: GPT = llm.model
model.set_kv_cache(batch_size=1, max_seq_length=50, device="cuda:0")


tok = next_token(model, input_pos, x)
print("Next Token:", tok)

tok = next_token(model, input_pos, x)