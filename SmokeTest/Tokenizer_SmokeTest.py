import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm

from Models.LatentGPT import LatentGPT
from Pretrain.Datasets   import Datasets
from torch.utils.data       import DataLoader

BATCH_SIZE  = 8
MAX_SEQ_LEN = 256
LR          = 1e-4
NUM_STEPS   = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(r"C:\junha\Git\BFG_2B\Tokenizer\spm_kowiki.model")

train_ds = Datasets(r"/Datasets/train.txt", tokenizer, MAX_SEQ_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

model = LatentGPT(
    vocab_size=tokenizer.GetPieceSize(),
    max_seq_len=MAX_SEQ_LEN,
    embed_dim=2688, latent_dim=336,
    mlp_dim=6912, num_layers=24,
    dropout=0.1
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn   = nn.CrossEntropyLoss(ignore_index=-100)

batch_iter = iter(train_loader)
inputs, labels = next(batch_iter)
inputs, labels = inputs.to(device), labels.to(device)

losses = []
model.train()
for step in range(1, NUM_STEPS + 1):
    optimizer.zero_grad()
    logits = model(inputs)  # (B, L, V)
    # CrossEntropyLoss expects (B*L, V) vs (B*L,)
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    print(f"Step {step:2d} | loss: {loss.item():.4f}")

print("Loss progression:", losses)
