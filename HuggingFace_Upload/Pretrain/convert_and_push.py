# convert_and_push.py

import os
import shutil
import json
import torch
import sentencepiece as spm
from configuration_raptor import RaptorConfig
from modeling_raptor import RaptorModel

CKPT = r"C:\junha\Git\BFG_2B\Checkpoints\KoRapter150M_Kowiki_AIHub_lr_1e_3\model_epoch_4.pt"
SPM_MODEL = "tokenizer.model"
OUT_DIR = "raptor-hf"  # HF 포맷으로 저장될 디렉터리
# REPO_ID 등 업로드 관련 코드는 전부 제거

# 1) SentencePiece vocab size 로드
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(SPM_MODEL)
VOCAB_SIZE = tokenizer.GetPieceSize()
print(f"[Info] Loaded SentencePiece vocab size: {VOCAB_SIZE}")

# hyperparams
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

# 2) Config 생성
cfg = RaptorConfig(
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
)

cfg.architectures = ["RaptorModel"]

# 3) 모델 생성 및 가중치 로드
model = RaptorModel(cfg)
state = torch.load(CKPT, map_location="cpu")
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"[Info] missing keys: {missing}")
print(f"[Info] unexpected keys: {unexpected}")

# 4) HF 포맷 저장 (모델 + config.json 생성)
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR)  # → config.json, pytorch_model.bin or model.safetensors 생성
cfg.save_pretrained(OUT_DIR)
shutil.copy("configuration_raptor.py", os.path.join(OUT_DIR, "configuration_raptor.py"))
shutil.copy("modeling_raptor.py", os.path.join(OUT_DIR, "modeling_raptor.py"))
# 빈 __init__.py 생성
open(os.path.join(OUT_DIR, "__init__.py"), "w").close()

# 5) 토크나이저 파일 복사 및 메타 생성
shutil.copy(SPM_MODEL, os.path.join(OUT_DIR, "tokenizer.model"))

tokenizer_config = {
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>"
}
with open(os.path.join(OUT_DIR, "tokenizer_config.json"), "w", encoding="utf-8") as f:
    json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

print(f"[Info] Saved everything to {OUT_DIR}:")
print("  - config.json")
print("  - pytorch_model.bin (or model.safetensors)")
print("  - tokenizer.model")
print("  - tokenizer_config.json")
