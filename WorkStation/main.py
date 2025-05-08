import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sentencepiece as spm
from tqdm.auto import tqdm

from Datasets import Datasets
from Models.LatentMoE import LatentMoE
from Train_Step import train_step
from Test_Step import test_step


def main():
    # ---------- 환경 ----------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True  # 성능 최적화

    # ---------- 하이퍼파라미터 ----------
    BATCH_SIZE = 1
    STRIDE = 128
    NUM_WORKERS = 4
    NUM_EPOCHS = 5
    LR = 1e-4
    ACCUM_STEPS = 8

    MAX_SEQ_LEN = 512
    NUM_HEADS = 16
    EMBED_DIM = 1024
    LATENT_DIM = 256
    MLP_DIM = 4096
    NUM_LAYERS = 16
    DROPOUT = 0.1
    NUM_EXPERTS = 6
    EXPERTS_PER_TOKEN = 2
    BALANCE_LOSS_WGT = 0.01

    # ---------- 토크나이저 ----------
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(r"C:\junha\Git\BFG_2B\Tokenizer\spm_kowiki.model")
    VOCAB_SIZE = tokenizer.GetPieceSize()

    # ---------- 데이터셋 ----------
    train_dataset = Datasets(
        r"C:\junha\Git\BFG_2B\Datasets\train.txt",
        tokenizer,
        max_seq_len=MAX_SEQ_LEN,
        stride=STRIDE,
    )
    val_dataset = Datasets(
        r"C:\junha\Git\BFG_2B\Datasets\val.txt",
        tokenizer,
        max_seq_len=MAX_SEQ_LEN,
        stride=STRIDE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # 분산 Sampler 대신 shuffle
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ---------- 모델 ----------
    model = LatentMoE(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        embed_dim=EMBED_DIM,
        latent_dim=LATENT_DIM,
        mlp_dim=MLP_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_heads=NUM_HEADS,
        num_experts=NUM_EXPERTS,
        experts_per_token=EXPERTS_PER_TOKEN,
        balance_loss_weight=BALANCE_LOSS_WGT,
    ).to(device)
    # weight tying
    model.lm_head.weight = model.token_embedding.weight

    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # ---------- 학습 ----------
    ckpt_dir = r"C:\junha\BFG_2B\Checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    epoch_iter = tqdm(range(1, NUM_EPOCHS + 1), desc="Epochs")

    for epoch in epoch_iter:
        train_loss, train_acc = train_step(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            accumulation_steps=ACCUM_STEPS,
            use_amp=True,
        )

        val_loss, val_acc, val_f1 = test_step(
            model,
            val_loader,
            loss_fn,
            device,
            use_amp=True,
        )

        epoch_iter.set_postfix({
            "Train Loss": f"{train_loss:.3f}",
            "Val Loss": f"{val_loss:.3f}",
            "Val Acc": f"{val_acc * 100:.2f}%",
        })

        torch.cuda.empty_cache()

        ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    main()
