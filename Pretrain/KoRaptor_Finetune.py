import os
import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset, concatenate_datasets
from peft import get_peft_model, LoraConfig, TaskType
from Models.LatentMoE import LatentMoE

device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 256
BATCH_SIZE = 4
GRAD_ACC_STEPS = 8
NUM_EPOCHS = 2
LEARNING_RATE = 1e-4
SAVE_STEPS = 500
OUTPUT_DIR = "./lora-finetuned-latentmoe"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sp = spm.SentencePieceProcessor()
sp.Load(r"C:\junha\Git\BFG_2B\Tokenizer\spm_kowiki.model")
PAD_ID = sp.PieceToId("<pad>")
EOS_ID = sp.PieceToId("</s>")

ds1 = load_dataset("beomi/KoAlpaca-v1.1a")["train"]
ds2 = load_dataset("MarkrAI/KoCommercial-Dataset", split="train")
raw_ds = concatenate_datasets([ds1, ds2])

def encode_fn(example):
    text = example["instruction"]
    if example.get("input"):
        text += "\n" + example["input"]
    ids = sp.EncodeAsIds(text)
    ids = ids[:MAX_SEQ_LEN-1] + [EOS_ID]
    pad_len = MAX_SEQ_LEN - len(ids)
    ids += [PAD_ID] * pad_len
    labs = sp.EncodeAsIds(example["output"])
    labs = labs[:MAX_SEQ_LEN//2-1] + [EOS_ID]
    pad_l = MAX_SEQ_LEN//2 - len(labs)
    labs += [PAD_ID] * pad_l
    return {"input_ids": ids, "labels": labs}

processed = raw_ds.map(encode_fn, remove_columns=raw_ds.column_names)
processed.set_format(type="torch", columns=["input_ids","labels"])

class SimpleDataset(Dataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        item = self.ds[i]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": (item["input_ids"] != PAD_ID).long(),
            "labels": item["labels"]
        }

train_ds = SimpleDataset(processed)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = LatentMoE(
    vocab_size=sp.GetPieceSize(),
    max_seq_len=MAX_SEQ_LEN,
    embed_dim=640,
    latent_dim=160,
    mlp_dim=1536,
    num_layers=8,
    dropout=0.1,
    num_heads=8,
    num_experts=6,
    experts_per_token=2,
    balance_loss_weight=0.01,
).to(device)
model = model.half()

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=[
        "q_proj",
        "dkv_proj",
        "up_proj_k",
        "up_proj_v",
        "out_proj",
        "fc1",
        "fc2",
        "lm_head",
    ],)
model = get_peft_model(model, lora_config)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = (len(train_loader) // GRAD_ACC_STEPS) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps
)

scaler = torch.cuda.amp.GradScaler()
global_step = 0

for epoch in range(NUM_EPOCHS):
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRAD_ACC_STEPS
        scaler.scale(loss).backward()
        if (step + 1) % GRAD_ACC_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            if global_step % SAVE_STEPS == 0:
                save_path = os.path.join(OUTPUT_DIR, f"checkpoint-step-{global_step}")
                model.save_pretrained(save_path)
                sp.Save(os.path.join(save_path, "spm.model"))
    save_path = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
    model.save_pretrained(save_path)
    sp.Save(os.path.join(save_path, "spm.model"))

model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
sp.Save(os.path.join(OUTPUT_DIR, "final_spm.model"))
