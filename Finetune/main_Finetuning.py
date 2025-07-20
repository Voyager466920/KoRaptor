import os
import torch
import sentencepiece as spm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import shutil


from Finetune.LatentMoE import LatentMoE

# ========= Configurations =========
PRETRAINED_MODEL_PATH = r"C:\junha\Git\BFG_2B\Checkpoints\KoRapter150M_Kowiki_AIHub_lr_1e_3\model_epoch_4.pt"
SPM_MODEL_PATH = r"C:\junha\Git\BFG_2B\Tokenizer\spm_kowiki.model"
MAX_SEQ_LEN = 256
BATCH_SIZE = 8
GRAD_ACC_STEPS = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
SAVE_STEPS = 1000
OUTPUT_DIR = "./lora_finetuned_koVast"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= Tokenizer =========
sp = spm.SentencePieceProcessor()
sp.Load(SPM_MODEL_PATH)
PAD_ID = sp.PieceToId("<pad>")
EOS_ID = sp.PieceToId("</s>")

# ========= Dataset =========
ds = load_dataset("maywell/koVast", split="train")

def encode_fn(example):
    conv = example["conversations"]
    text = ""
    for turn in conv:
        text += turn["value"] + " "
    ids = sp.EncodeAsIds(text.strip())
    ids = ids[: MAX_SEQ_LEN - 1] + [EOS_ID]
    pad_len = MAX_SEQ_LEN - len(ids)
    ids += [PAD_ID] * pad_len
    input_ids = ids[:-1]
    labels = ids[1:]
    return {"input_ids": input_ids, "labels": labels}

processed = ds.map(encode_fn, remove_columns=ds.column_names)
processed.set_format(type="torch", columns=["input_ids", "labels"])
train_loader = DataLoader(processed, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ========= Model & LoRA =========
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
    pad_token_id=PAD_ID,
    eos_token_id=EOS_ID
)
model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, weights_only=True))
model.to(device).half()

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=[
        "q_proj", "dkv_proj", "up_proj_k", "up_proj_v", "out_proj",
        "fc1", "fc2", "moe.gate", "lm_head",
    ],
)
model = get_peft_model(model, lora_config)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
steps_per_epoch = len(train_loader) // GRAD_ACC_STEPS
total_steps = steps_per_epoch * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps,
)

scaler = torch.cuda.amp.GradScaler()
global_step = 0

# ========= Training Loop (with tqdm) =========
for epoch in range(NUM_EPOCHS):
    for step, batch in enumerate(
        tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
            unit="batch"
        ),
        start=1
    ):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = (input_ids != PAD_ID).long().to(device)

        with torch.cuda.amp.autocast():
            logits, balance_loss = model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)
            ce_loss = loss_fn(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
            loss = ce_loss + balance_loss * model.balance_loss_weight
            loss = loss / GRAD_ACC_STEPS

        scaler.scale(loss).backward()
        if (step % GRAD_ACC_STEPS) == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            if global_step % SAVE_STEPS == 0:
                ckpt = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}.pt")
                model.save_pretrained(ckpt)

    epoch_ckpt = os.path.join(OUTPUT_DIR, f"epoch-{epoch+1}")
    model.save_pretrained(epoch_ckpt)

model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
shutil.copy(SPM_MODEL_PATH, os.path.join(OUTPUT_DIR, "final_spm.model"))
