import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sentencepiece as spm
from tqdm.auto import tqdm

from Models.LatentGPT import LatentGPT
from WorkStation.Datasets import Datasets
from WorkStation.Train_Step import train_step
from WorkStation.Test_Step import test_step

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE   = 64
MAX_SEQ_LEN  = 256
EMBED_DIM    = 2688
LATENT_DIM   = 336
MLP_DIM      = 6912
NUM_LAYERS   = 24
DROPOUT      = 0.1
NUM_EPOCHS   = 100
LR           = 1e-4
ACCUM_STEPS  = 4

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(r"C:\junha\Git\BFG_2B\Tokenizer\spm_kowiki.model")
VOCAB_SIZE = tokenizer.GetPieceSize()

def main():
    train_dataset = Datasets(r"C:\junha\Git\BFG_2B\Datasets\train.txt", tokenizer, max_seq_len=MAX_SEQ_LEN)
    val_dataset   = Datasets(r"C:\junha\Git\BFG_2B\Datasets\val.txt", tokenizer, max_seq_len=MAX_SEQ_LEN)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    model = LatentGPT(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        embed_dim=EMBED_DIM,
        latent_dim=LATENT_DIM,
        mlp_dim=MLP_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    loss_fn   = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in tqdm(range(NUM_EPOCHS)):
        train_loss, train_acc = train_step(
            model, train_dataloader, loss_fn, optimizer, device,
            accumulation_steps=ACCUM_STEPS
        )
        val_loss, val_acc, val_f1 = test_step(
            model, val_dataloader, loss_fn, device
        )

        print(f"Epoch {epoch}/{NUM_EPOCHS+1}:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f}\n")

if __name__ == "__main__":
    main()