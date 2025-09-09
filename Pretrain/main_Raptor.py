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
from Pretrain.StreamingDataset import StreamingDataset


def zip_alternate(*iters):
    its = [iter(it) for it in iters]
    while its:
        for it in list(its):
            try:
                yield next(it)
            except StopIteration:
                its.remove(it)


class ChunkStream(IterableDataset):
    def __init__(self, raw_iter, tokenizer, max_seq_len, stride):
        self.raw_iter = raw_iter
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride

    def __iter__(self):
        for chunk in StreamingDataset(
                self.raw_iter,
                self.tokenizer,
                max_seq_len=self.max_seq_len,
                stride=self.stride
        ):
            yield chunk


class ShuffleStream(IterableDataset):
    def __init__(self, map_ds, tokenizer, max_seq_len, stride):
        self.map_ds = map_ds
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        ds_shuf = self.map_ds.shuffle(seed=self.epoch)
        raw_iter = ds_shuf.to_iterable_dataset()
        return iter(ChunkStream(
            raw_iter,
            self.tokenizer,
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

    # Hyperparameters
    BATCH_SIZE = 150
    STRIDE = 256
    NUM_WORKERS = 0
    NUM_EPOCHS = 20
    LR = 5e-4
    ACCUM_STEPS = 1

    MAX_SEQ_LEN = 256
    NUM_HEADS = 8
    EMBED_DIM = 640
    LATENT_DIM = 160
    MLP_DIM = 1536
    NUM_LAYERS = 8
    DROPOUT = 0.1
    NUM_EXPERTS = 6
    EXPERTS_PER_TOKEN = 2
    BALANCE_LOSS_WEIGHT = 0.01

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(r"C:\junha\Git\BFG_2B\Tokenizer\spm_kowiki.model")
    VOCAB_SIZE = tokenizer.GetPieceSize()

    WIKI2_TRAIN = r"C:\junha\Datasets\WikiText2\train"
    WIKI2_VAL = r"C:\junha\Datasets\WikiText2\val"
    WIKI103_TRAIN = r"C:\junha\Datasets\WikiText103\train"
    WIKI103_VAL = r"C:\junha\Datasets\WikiText103\val"
    OWT2_TRAIN = r"C:\junha\Datasets\OpenWebText2\train"
    OWT2_VAL = r"C:\junha\Datasets\OpenWebText2\val"
    BOOK_TRAIN = r"C:\junha\Datasets\BookCorpus\train"
    BOOK_VAL = r"C:\junha\Datasets\BookCorpus\val"

    wiki2_train_map = load_from_disk(WIKI2_TRAIN)
    wiki2_val_map = load_from_disk(WIKI2_VAL)
    wiki103_train_map = load_from_disk(WIKI103_TRAIN)
    wiki103_val_map = load_from_disk(WIKI103_VAL)
    owt2_train_map = load_from_disk(OWT2_TRAIN)
    owt2_val_map = load_from_disk(OWT2_VAL)
    book_train_map = load_from_disk(BOOK_TRAIN)
    book_val_map = load_from_disk(BOOK_VAL)

    wiki2_train_stream = ShuffleStream(wiki2_train_map, tokenizer, MAX_SEQ_LEN, STRIDE)
    wiki2_val_stream = ShuffleStream(wiki2_val_map, tokenizer, MAX_SEQ_LEN, STRIDE)
    wiki103_train_stream = ShuffleStream(wiki103_train_map, tokenizer, MAX_SEQ_LEN, STRIDE)
    wiki103_val_stream = ShuffleStream(wiki103_val_map, tokenizer, MAX_SEQ_LEN, STRIDE)
    owt2_train_stream = ShuffleStream(owt2_train_map, tokenizer, MAX_SEQ_LEN, STRIDE)
    owt2_val_stream = ShuffleStream(owt2_val_map, tokenizer, MAX_SEQ_LEN, STRIDE)
    book_train_stream = ShuffleStream(book_train_map, tokenizer, MAX_SEQ_LEN, STRIDE)
    book_val_stream = ShuffleStream(book_val_map, tokenizer, MAX_SEQ_LEN, STRIDE)

    train_dataset = CombinedDataset(wiki2_train_stream, wiki103_train_stream, owt2_train_stream, book_train_stream)
    val_dataset = CombinedDataset(wiki2_val_stream, wiki103_val_stream, owt2_val_stream, book_val_stream)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

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

    ckpt_dir = r"C:\junha\Git\BFG_2B\Checkpoints\Rapter150M_Wiki2_103_Book_Refined"
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
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt"))

if __name__ == "__main__":
    main()
