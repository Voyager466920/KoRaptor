import os
from itertools import chain
from datasets import load_from_disk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sentencepiece as spm
from tqdm.auto import tqdm

from Models.LatentMoE import LatentMoE
from Train_Step import train_step
from Test_Step import test_step
from WorkStation.StreamingDataset import StreamingDataset


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

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(r"C:\junha\Git\BFG_2B\Tokenizer\spm_kowiki.model")
    VOCAB_SIZE = tokenizer.GetPieceSize()

    raw_train_map = load_from_disk(r"C:\junha\Datasets\KoWiki_TrainVal\train")
    raw_val_map = load_from_disk(r"C:\junha\Datasets\KoWiki_TrainVal\val")
    raw_wiki_train = raw_train_map.map(lambda e: {"text": e["sentence"]}).to_iterable_dataset()
    raw_wiki_val = raw_val_map.map(lambda e: {"text": e["sentence"]}).to_iterable_dataset()

    raw_train_korean = load_from_disk(r"C:\junha\Datasets\KoreanText\Train").map(
        lambda e: {"text": e["text"]}).to_iterable_dataset()
    raw_test_korean = load_from_disk(r"C:\junha\Datasets\KoreanText\Test").map(
        lambda e: {"text": e["text"]}).to_iterable_dataset()

    raw_train = chain(raw_wiki_train, raw_train_korean)
    raw_val = raw_wiki_val
    raw_test = chain(raw_wiki_val, raw_test_korean)

    train_dataset = StreamingDataset(raw_train, tokenizer, max_seq_len=MAX_SEQ_LEN, stride=STRIDE)
    val_dataset = StreamingDataset(raw_val, tokenizer, max_seq_len=MAX_SEQ_LEN, stride=STRIDE)
    test_dataset = StreamingDataset(raw_test, tokenizer, max_seq_len=MAX_SEQ_LEN, stride=STRIDE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

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

    ckpt_dir = r"/Checkpoints/Rapter72M_KoWiki"
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Epochs"):
        train_ppl = train_step(model, train_loader, loss_fn, optimizer, device, accumulation_steps=ACCUM_STEPS,
                               use_amp=True)
        val_ppl, val_acc = test_step(model, val_loader, loss_fn, device, use_amp=True)
        scheduler.step()
        torch.cuda.empty_cache()
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"KoWiki_72M_epoch_{epoch}.pt"))

    test_ppl, test_acc = test_step(model, test_loader, loss_fn, device, use_amp=True)
    print(f"Test PPL: {test_ppl:.1f}  Test Acc: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
