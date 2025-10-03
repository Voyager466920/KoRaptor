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

    kowiki_train_map = load_from_disk(r"C:\junha\Datasets\KoWiki_TrainVal\train")
    kowiki_val_map = load_from_disk(r"C:\junha\Datasets\KoWiki_TrainVal\val")
    koreantext_train_map = load_from_disk(r"C:\junha\Datasets\KoreanText\train")
    koreantext_val_map = load_from_disk(r"C:\junha\Datasets\KoreanText\val")
    interview_train_map = load_from_disk(r"C:\junha\Datasets\KoRaptor_Pretrain\Interview\train")
    interview_val_map = load_from_disk(r"C:\junha\Datasets\KoRaptor_Pretrain\Interview\val")
    news_train_map = load_from_disk(r"C:\junha\Datasets\KoRaptor_Pretrain\KoRaptor_news\train")
    news_val_map = load_from_disk(r"C:\junha\Datasets\KoRaptor_Pretrain\KoRaptor_news\val")

    kowiki_train_stream = ShuffleStream(kowiki_train_map, tokenizer, MAX_SEQ_LEN, STRIDE)
    koreantext_train_stream = ShuffleStream(koreantext_train_map, tokenizer, MAX_SEQ_LEN, STRIDE)
    interview_train_stream = ShuffleStream(interview_train_map, tokenizer, MAX_SEQ_LEN, STRIDE)
    news_train_stream = ShuffleStream(news_train_map, tokenizer, MAX_SEQ_LEN, STRIDE)

    kowiki_val_stream = ShuffleStream(kowiki_val_map, tokenizer, MAX_SEQ_LEN, STRIDE)
    koreantext_val_stream = ShuffleStream(koreantext_val_map, tokenizer, MAX_SEQ_LEN, STRIDE)
    interview_val_stream = ShuffleStream(interview_val_map, tokenizer, MAX_SEQ_LEN, STRIDE)
    news_val_stream = ShuffleStream(news_val_map, tokenizer, MAX_SEQ_LEN, STRIDE)

    train_dataset = CombinedDataset(kowiki_train_stream, koreantext_train_stream, interview_train_stream, news_train_stream)
    val_dataset = CombinedDataset(kowiki_val_stream, koreantext_val_stream, interview_val_stream, news_val_stream)

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

    ckpt_dir = r"/Checkpoints/KoRapter150M_Kowiki_251004"
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
