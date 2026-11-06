import torch
from tokenizer import tokenizer

with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()

tok = tokenizer()

try:
    tok.load("tokenizer.json")
    data = torch.load("data.pt")
except FileNotFoundError:
    print("Training tokenizer and encoding data...")
    tok.train(text, 500)
    tok.save("tokenizer.json")
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    torch.save(data, "data.pt")

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch("train")
print("inputs:")
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)
