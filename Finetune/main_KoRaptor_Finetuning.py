import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sentencepiece as spm
from peft import LoraConfig, get_peft_model, TaskType
from tqdm.auto import tqdm

from Finetune.KoChatDataset import KoChatDataset
from Finetune.LatentMoE import LatentMoE, LatentMoEShim
from Pretrain.Test_Step import test_step
from Pretrain.Train_Step import train_step



def collate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "labels": labels}


def main():
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    MAX_SEQ_LEN = 256
    BATCH_SIZE = 150
    LR = 1e-3
    NUM_EPOCHS = 10

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(r"C:\junha\Git\BFG_2B\Tokenizer\spm_kowiki.model")

    train_dataset = KoChatDataset(
        r"C:\junha\Datasets\KoRaptor_FineTuning\train_simplified.jsonl",
        tokenizer, MAX_SEQ_LEN
    )
    val_dataset = KoChatDataset(
        r"C:\junha\Datasets\KoRaptor_FineTuning\val_simplified.jsonl",
        tokenizer, MAX_SEQ_LEN
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    base_model = LatentMoE(
        vocab_size=tokenizer.GetPieceSize(),
        max_seq_len=MAX_SEQ_LEN,
        embed_dim=640, latent_dim=160, mlp_dim=1536,
        num_layers=8, num_heads=8, dropout=0.1,
        num_experts=6, experts_per_token=2, balance_loss_weight=0.01,
    )
    base_model.load_state_dict(torch.load(
        r"/Checkpoints/A_HuggingFace_KoRapter150M_Kowiki_AIHub_lr_1e_3\model_epoch_4.pt"
    ))
    base_model.to(device)

    shim = LatentMoEShim(base_model).to(device)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=[
            "q_proj", "dkv_proj", "up_proj_k", "up_proj_v",
            "out_proj", "fc1", "fc2",
        ],
    )
    model = get_peft_model(shim, peft_config)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
        model.train()
        train_ppl = train_step(
            model, train_loader, loss_fn, optimizer, device,
            accumulation_steps=1, use_amp=True
        )

        model.eval()
        with torch.no_grad():
            val_ppl, val_acc = test_step(
                model, val_loader, loss_fn, device, use_amp=True
            )

        scheduler.step()
        print(f"[Epoch {epoch}] train_ppl={train_ppl:.2f}, "
              f"val_ppl={val_ppl:.2f}, val_acc={val_acc * 100:.2f}%")

        # LoRA 체크포인트 저장
        save_dir = f"C:/junha/Git/BFG_2B/Checkpoints/lora_checkpoint/epoch_{epoch}"
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
