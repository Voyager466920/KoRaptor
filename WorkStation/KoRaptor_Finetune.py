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


def collate_fn(batch):
    input_ids = nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch],
        batch_first=True,
        padding_value=0
    )
    labels = nn.utils.rnn.pad_sequence(
        [b["target_ids"] for b in batch],
        batch_first=True,
        padding_value=-100
    )
    return {"input_ids": input_ids, "labels": labels}


class QAChunkStream(IterableDataset):
    def __init__(self, hf_split, tokenizer, max_seq_len, stride):
        self.hf_split = hf_split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride

    def __iter__(self):
        for ex in self.hf_split:
            text = f"질문: {ex['question']}  문맥: {ex['context']}  답변: {ex['answers']['text'][0]}"
            ids = self.tokenizer.EncodeAsIds(text)
            for i in range(0, len(ids), self.stride):
                chunk = ids[i: i + self.max_seq_len]
                if len(chunk) < 2:
                    continue
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                target_ids = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "target_ids": target_ids}


def main():
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    BATCH_SIZE = 150
    STRIDE = 256
    NUM_WORKERS = 0
    NUM_EPOCHS = 20
    LR = 3e-4
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

    train_split = load_from_disk(r"C:\junha\Datasets\KorQuAD\v1.0\train")
    val_split = load_from_disk(r"C:\junha\Datasets\KorQuAD\v1.0\val")

    train_stream = QAChunkStream(train_split, tokenizer, MAX_SEQ_LEN, STRIDE)
    val_stream = QAChunkStream(val_split, tokenizer, MAX_SEQ_LEN, STRIDE)

    train_loader = DataLoader(
        train_stream,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_stream,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

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
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    ckpt_dir = r"C:\junha\Git\BFG_2B\Checkpoints\KoRapter_KorQuAD_Finetuned"
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Epochs"):
        train_ppl = train_step(
            model, train_loader, loss_fn, optimizer, device,
            accumulation_steps=ACCUM_STEPS, use_amp=True
        )
        val_ppl, _ = test_step(
            model, val_loader, loss_fn, device, use_amp=True
        )
        scheduler.step()
        tqdm.write(f"[Epoch {epoch}] Train PPL: {train_ppl:.2f}  Val PPL: {val_ppl:.2f}")
        torch.save(
            model.state_dict(),
            os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt")
        )


if __name__ == "__main__":
    main()
