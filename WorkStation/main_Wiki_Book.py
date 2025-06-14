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
                stride=self.stride
        ):
            L = len(chunk["input_ids"])
            if total + L > self.token_limit:
                break
            total += L
            yield chunk

class ShuffleStream(IterableDataset):
    def __init__(self, map_ds, tokenizer, token_limit, max_seq_len, stride):
        self.map_ds = map_ds
        self.tokenizer = tokenizer
        self.token_limit = token_limit
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        ds_shuf = self.map_ds.shuffle(seed=self.epoch)
        raw_iter = ds_shuf.to_iterable_dataset()
        return iter(TokenLimitStream(
            raw_iter,
            self.tokenizer,
            token_limit=self.token_limit,
            max_seq_len=self.max_seq_len,
            stride=self.stride
        ))

class CombinedDataset(IterableDataset):
    def __init__(self, *streams):
        self.streams = streams

    def set_epoch(self, epoch):
        for s in self.streams:
            if hasattr(s, 'set_epoch'):
                s.set_epoch(epoch)

    def __iter__(self):
        return zip_alternate(*self.streams)


def main():
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    BATCH_SIZE = 180
    STRIDE = 256
    NUM_WORKERS = 0
    NUM_EPOCHS = 20
    LR = 5e-4
    ACCUM_STEPS = 1

    MAX_SEQ_LEN = 256
    NUM_HEADS = 8
    EMBED_DIM = 512
    LATENT_DIM = 128
    MLP_DIM = 1024
    NUM_LAYERS = 6
    DROPOUT = 0.1
    NUM_EXPERTS = 5
    EXPERTS_PER_TOKEN = 1
    BALANCE_LOSS_WEIGHT = 0.01

    BOOK_TOKEN_LIMIT_TRAIN = 103_000_000
    BOOK_TOKEN_LIMIT_VAL = 245_000

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(r"C:\junha\Git\BFG_2B\Tokenizer\spm_bc.model")
    VOCAB_SIZE = tokenizer.GetPieceSize()

    book_train_map = load_from_disk(r"C:\junha\Datasets\BookCorpus\train")
    book_val_map = load_from_disk(r"C:\junha\Datasets\BookCorpus\val")
    wiki_train_map = load_from_disk(r"C:\junha\Datasets\WikiText103\train")
    wiki_val_map = load_from_disk(r"C:\junha\Datasets\WikiText103\val")

    book_train_stream = ShuffleStream(book_train_map, tokenizer, BOOK_TOKEN_LIMIT_TRAIN, MAX_SEQ_LEN, STRIDE)
    wiki_train_stream = ShuffleStream(wiki_train_map, tokenizer, BOOK_TOKEN_LIMIT_TRAIN, MAX_SEQ_LEN, STRIDE)
    book_val_stream = ShuffleStream(book_val_map, tokenizer, BOOK_TOKEN_LIMIT_VAL, MAX_SEQ_LEN, STRIDE)
    wiki_val_stream = ShuffleStream(wiki_val_map, tokenizer, BOOK_TOKEN_LIMIT_VAL, MAX_SEQ_LEN, STRIDE)

    train_dataset = CombinedDataset(book_train_stream, wiki_train_stream)
    val_dataset = CombinedDataset(book_val_stream, wiki_val_stream)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")

    ckpt_dir = r"C:\junha\Git\BFG_2B\Checkpoints\BFG72M_Wiki_Book"
    os.makedirs(ckpt_dir, exist_ok=True)

    epoch_iter = tqdm(range(1, NUM_EPOCHS + 1), desc="Epochs")
    for epoch in epoch_iter:
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(epoch)

        train_ppl = train_step(
            model, train_dataloader, loss_fn, optimizer, device,
            accumulation_steps=ACCUM_STEPS, use_amp=True
        )
        val_ppl, val_acc = test_step(
            model, val_dataloader, loss_fn, device, use_amp=True
        )

        scheduler.step()

        epoch_iter.set_postfix({
            "Train PPL": f"{train_ppl:.1f}",
            "Val PPL": f"{val_ppl:.1f}",
            "Val Acc": f"{val_acc * 100:.2f}%"
        })

        torch.cuda.empty_cache()
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"72M_model_epoch_{epoch}.pt"))

if __name__ == "__main__":
    main()
