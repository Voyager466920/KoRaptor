import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, distributed
import sentencepiece as spm
from tqdm.auto import tqdm

from Models.LatentGPT import LatentGPT
from Datasets import Datasets
from Train_Step import train_step
from Test_Step import test_step

def main():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    torch.backends.cudnn.benchmark = True

    BATCH_SIZE = 1
    MAX_SEQ_LEN = 256
    NUM_HEADS = 32
    EMBED_DIM = 2048
    LATENT_DIM = 336
    MLP_DIM = 4096
    NUM_LAYERS = 24
    DROPOUT = 0.1
    NUM_EPOCHS = 100
    LR = 1e-4
    ACCUM_STEPS = 8
    STRIDE = 128
    NUM_WORKERS = 4

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load("/home/junha/BFG_2B/Tokenizer/spm_kowiki.model")
    VOCAB_SIZE = tokenizer.GetPieceSize()
    Checkpoint="/home/junha/BFG_2B/Checkpoints"

    train_dataset = Datasets(
        "/home/junha/BFG_2B/Datasets/train.txt",
        tokenizer,
        max_seq_len=MAX_SEQ_LEN,
        stride=STRIDE
    )
    val_dataset = Datasets(
        "/home/junha/BFG_2B/Datasets/val.txt",
        tokenizer,
        max_seq_len=MAX_SEQ_LEN,
        stride=STRIDE
    )

    train_sampler = distributed.DistributedSampler(train_dataset)
    val_sampler = distributed.DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=NUM_WORKERS
    )

    model = LatentGPT(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        embed_dim=EMBED_DIM,
        latent_dim=LATENT_DIM,
        mlp_dim=MLP_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_heads=NUM_HEADS
    ).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    if local_rank == 0:
        os.makedirs("Checkpoint", exist_ok=True)
        epoch_iter = tqdm(range(1, NUM_EPOCHS + 1), desc="Epochs")
    else:
        epoch_iter = range(1, NUM_EPOCHS + 1)

    for epoch in epoch_iter:
        train_sampler.set_epoch(epoch)
        train_loss, train_acc = train_step(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            accumulation_steps=ACCUM_STEPS,
            use_amp=True
        )
        val_loss, val_acc, val_f1 = test_step(
            model,
            val_loader,
            loss_fn,
            device,
            use_amp=True
        )

        if local_rank == 0:
            epoch_iter.set_postfix({
                "Train Loss": f"{train_loss:.3f}",
                "Val Loss":   f"{val_loss:.3f}",
                "Val Acc":    f"{val_acc*100:.2f}%"
            })
            torch.cuda.empty_cache()
            checkpoint_path = os.path.join(Checkpoint, f"model_epoch_{epoch}.pt")
            torch.save(model.module.state_dict(), checkpoint_path)

if __name__ == "__main__":
    main()
