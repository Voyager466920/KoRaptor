import json
import sentencepiece as spm
from HuggingFace_Upload.Finetune.LatentMoE_Config import LatentMoEShimConfig

# ─── 모델 구조 하이퍼파라미터 ────────────────────────────────────────────
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
# ────────────────────────────────────────────────────────────────────────

# 1) adaptor_config 불러오기
with open(r"/Checkpoints/KoRaptor150M_Chatbot_checkpoint\epoch_10\adapter_config.json", "r") as f:
    adaptor_cfg = json.load(f)


# 2) tokenizer 로드 & special token IDs 추출
def load_tokenizer(model_path: str):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp


TOKENIZER_PATH = r"C:\junha\Git\BFG_2B\Tokenizer\spm_kowiki.model"
tokenizer = load_tokenizer(TOKENIZER_PATH)
VOCAB_SIZE = tokenizer.get_piece_size()

# special tokens를 학습 시 정의한 이름으로 대체하세요
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

pad_id = tokenizer.piece_to_id(PAD_TOKEN)
bos_id = tokenizer.piece_to_id(BOS_TOKEN)
eos_id = tokenizer.piece_to_id(EOS_TOKEN)

print(f"Vocab size: {VOCAB_SIZE}, pad_id: {pad_id}, bos_id: {bos_id}, eos_id: {eos_id}")

# 3) Hugging Face 형식의 config 객체 생성
config = LatentMoEShimConfig(
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
    adaptor_config=adaptor_cfg,
    pad_token_id=pad_id,
    bos_token_id=bos_id,
    eos_token_id=eos_id,
)

# 4) config.json으로 저장
config.save_pretrained("hf_repo")
print("✅ config.json 생성 완료: hf_repo/config.json")
