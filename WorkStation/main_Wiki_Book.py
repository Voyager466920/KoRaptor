import os
from datasets import load_from_disk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import sentencepiece as spm
from tqdm.auto import tqdm

from Models.LatentMoE import LatentMoE
from Train_Step import train_step
from Test_Step import test_step
from WorkStation.StreamingDataset import StreamingDataset


def zip_alternate(*iters):
    its = [iter(it) for it in iters]
    while its:
        for it in list(its):
            try:
                yield next(it)
            except StopIteration:
                its.remove(it)


class TokenLimitStream(IterableDataset):
    def __init__(self, raw_iter, tokenizer, token_limit, max_seq_len, stride):
        self.raw_iter = raw_iter
        self.tokenizer = tokenizer
        self.token_limit = token_limit
        self.max_seq_len = max_seq_len
        self.stride = stride

    def __iter__(self):
        total = 0
        for chunk in StreamingDataset(
                self.raw_iter,
                self.tokenizer,
                max_seq_len=self.max_seq_len,
                stride=self.stride,
        ):
            L = len(chunk["input_ids"])
            if total + L > self.token_limit:
                break
            total += L
            yield chunk


class IterableFromIterator(IterableDataset):
    def __init__(self, it): self.it = it

    def __iter__(self): return iter(self.it)


def main():
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ——— 하이퍼파라미터 ———
    BATCH_SIZE = 196
    STRIDE = 256
    NUM_WORKERS = 0
    NUM_EPOCHS = 15
    LR = 1e-4
    ACCUM_STEPS = 8

    MAX_SEQ_LEN = 256
    NUM_HEADS = 8
    EMBED_DIM = 512
    LATENT_DIM = 128
    MLP_DIM = 1024
    NUM_LAYERS = 6
    DROPOUT = 0.1
    NUM_EXPERTS = 5
    EXPERTS_PER_TOKEN = 1
    BALANCE_LOSS_WEIGHT = 0.01 # 72.2M

    BOOK_TOKEN_LIMIT_TRAIN = 103_000_000
    BOOK_TOKEN_LIMIT_VAL = 245_000

    # ——— 토크나이저 로드 ———
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(r"C:\junha\Git\BFG_2B\Tokenizer\spm_bc.model")
    VOCAB_SIZE = tokenizer.GetPieceSize()

    # ——— 데이터 로드 (map-style) ———
    book_train_map = load_from_disk(r"C:\junha\Datasets\BookCorpus\train")
    book_val_map = load_from_disk(r"C:\junha\Datasets\BookCorpus\val")
    wiki_train_map = load_from_disk(r"C:\junha\Datasets\WikiText103\train")
    wiki_val_map = load_from_disk(r"C:\junha\Datasets\WikiText103\val")

    # ——— 모델·옵티마이저 초기화 ———
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
        balance_loss_weight=BALANCE_LOSS_WEIGHT,
    ).to(device)
    model.lm_head.weight = model.token_embedding.weight

    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6,
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")

    ckpt_dir = r"C:\junha\Git\BFG_2B\Checkpoints\BFG72M_Wiki_Book"
    os.makedirs(ckpt_dir, exist_ok=True)

    epoch_iter = tqdm(range(1, NUM_EPOCHS + 1), desc="Epochs")
    for epoch in epoch_iter:
        book_train_shuf = book_train_map.shuffle(seed=epoch)
        wiki_train_shuf = wiki_train_map.shuffle(seed=epoch)
        book_val_shuf = book_val_map.shuffle(seed=epoch)
        wiki_val_shuf = wiki_val_map.shuffle(seed=epoch)

        book_train_iter = book_train_shuf.to_iterable_dataset()
        wiki_train_iter = wiki_train_shuf.to_iterable_dataset()
        book_val_iter = book_val_shuf.to_iterable_dataset()
        wiki_val_iter = wiki_val_shuf.to_iterable_dataset()

        book_train_lim = TokenLimitStream(
            book_train_iter, tokenizer,
            token_limit=BOOK_TOKEN_LIMIT_TRAIN,
            max_seq_len=MAX_SEQ_LEN,
            stride=STRIDE,
        )
        wiki_train_lim = TokenLimitStream(
            wiki_train_iter, tokenizer,
            token_limit=BOOK_TOKEN_LIMIT_TRAIN,
            max_seq_len=MAX_SEQ_LEN,
            stride=STRIDE,
        )
        book_val_lim = TokenLimitStream(
            book_val_iter, tokenizer,
            token_limit=BOOK_TOKEN_LIMIT_VAL,
            max_seq_len=MAX_SEQ_LEN,
            stride=STRIDE,
        )
        wiki_val_lim = TokenLimitStream(
            wiki_val_iter, tokenizer,
            token_limit=BOOK_TOKEN_LIMIT_VAL,
            max_seq_len=MAX_SEQ_LEN,
            stride=STRIDE,
        )

        train_dataset = IterableFromIterator(zip_alternate(book_train_lim, wiki_train_lim))
        val_dataset = IterableFromIterator(zip_alternate(book_val_lim, wiki_val_lim))
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        train_ppl, train_acc = train_step(
            model, train_dataloader, loss_fn, optimizer, device,
            accumulation_steps=ACCUM_STEPS, use_amp=True
        )
        val_ppl, val_acc, val_f1 = test_step(
            model, val_dataloader, loss_fn, device, use_amp=True
        )

        scheduler.step()

        epoch_iter.set_postfix({
            "Train PPL": f"{train_ppl:.1f}",
            "Val PPL": f"{val_ppl:.1f}",
            "Val Acc": f"{val_acc * 100:.2f}%",
            "Val F1": f"{val_f1:.4f}"
        })

        torch.cuda.empty_cache()
        torch.save(
            model.state_dict(),
            os.path.join(ckpt_dir, f"72M_model_epoch_{epoch}.pt")
        )


if __name__ == "__main__":
    main()
