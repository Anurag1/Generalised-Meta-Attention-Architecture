from marm import MARM
from transformers import AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = MARM(vocab_size=tokenizer.vocab_size).to(device)
model.load_state_dict(torch.load("marm_common_crawl.pt"))
model.eval()

def generate(prompt, max_len=50):
    tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    for _ in range(max_len):
        out = model(tokens)
        next_token = out["logits"][:, -1].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokenizer.decode(tokens[0])

print(generate("The meaning of intelligence is"))
