import os
import torch
import torch.nn.functional as F
import sentencepiece as spm
from collections import Counter
from peft import PeftModelForCausalLM
from Finetune.LatentMoE import LatentMoE

# ─── 설정 ─────────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPM_PATH   = r"C:\junha\Git\BFG_2B\Tokenizer\spm_kowiki.model"
CKPT_DIR   = "./lora_finetuned_koVast/final"
MAX_LEN    = 100
TEMPERATURE = 0.8
TOP_K       = 10
TOP_P       = 0.9
REPETITION_PENALTY = 1.2
NO_REPEAT_NGRAM_SIZE = 3
BEAM_WIDTH  = 3
LENGTH_PENALTY = 0.7

# ─── 토크나이저 & 모델 로드 ────────────────────────────────────────────────────
sp = spm.SentencePieceProcessor()
sp.Load(SPM_PATH)
PAD_ID = sp.PieceToId("<pad>")
EOS_ID = sp.PieceToId("</s>")

base = LatentMoE(
    vocab_size=sp.GetPieceSize(),
    max_seq_len=256,
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
model = PeftModelForCausalLM.from_pretrained(base, CKPT_DIR)
model.to(DEVICE).half().eval()

# ─── 공통 유틸 ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def enforce_no_repeat_ngram(logits, generated, n):
    if generated.size(1) < n:
        return logits
    prev = tuple(generated[0, -(n-1):].tolist())
    banned = set()
    for i in range(generated.size(1) - (n-1)):
        if tuple(generated[0, i:i+(n-1)].tolist()) == prev:
            banned.add(generated[0, i+(n-1)].item())
    for token in banned:
        logits[:, token] = -1e9
    return logits

def apply_repetition_penalty(logits, generated, penalty):
    counts = Counter(generated.view(-1).tolist())
    for tok, cnt in counts.items():
        if cnt > 1:
            logits[:, tok] /= penalty ** (cnt - 1)
    return logits

# ─── 샘플링 생성 ─────────────────────────────────────────────────────────────
@torch.no_grad()
def sample_sequence(input_ids):
    generated = input_ids
    for _ in range(MAX_LEN):
        logits, _ = model(input_ids=generated, attention_mask=(generated!=PAD_ID).long())
        next_logits = logits[:, -1, :] / TEMPERATURE

        if TOP_K > 0:
            v, _ = torch.topk(next_logits, TOP_K)
            next_logits[next_logits < v[:, -1].unsqueeze(-1)] = -1e9

        if TOP_P < 1.0:
            s_logits, s_idx = torch.sort(next_logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(s_logits, dim=-1), dim=-1)
            mask = cumprobs > TOP_P
            mask[..., 1:] = mask[..., :-1].clone()
            s_logits[mask] = -1e9
            next_logits = torch.zeros_like(next_logits).scatter(1, s_idx, s_logits)

        next_logits = apply_repetition_penalty(next_logits, generated, REPETITION_PENALTY)
        next_logits = enforce_no_repeat_ngram(next_logits, generated, NO_REPEAT_NGRAM_SIZE)

        probs = F.softmax(next_logits, dim=-1)
        if not torch.isfinite(probs).all() or probs.sum() <= 0:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=-1)
        if next_token.item() == EOS_ID:
            break

    return generated

# ─── 빔 서치 생성 ─────────────────────────────────────────────────────────────
@torch.no_grad()
def beam_search(input_ids):
    beams = [(input_ids, 0.0)]
    for _ in range(MAX_LEN):
        candidates = []
        for seq, score in beams:
            logits, _ = model(input_ids=seq, attention_mask=(seq!=PAD_ID).long())
            next_logits = logits[:, -1, :] / TEMPERATURE
            log_probs = F.log_softmax(next_logits, dim=-1).squeeze(0)

            topk_lp, topk_idx = torch.topk(log_probs, BEAM_WIDTH)
            for lp, idx in zip(topk_lp, topk_idx):
                new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                length = new_seq.size(1)
                adjusted = lp.item() / (length ** LENGTH_PENALTY)
                candidates.append((new_seq, score + adjusted))

        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:BEAM_WIDTH]
        if all(b[0][0, -1].item() == EOS_ID for b in beams):
            break

    return beams[0][0]

# ─── 실행 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    history = [
        "대학 입학 단계에서 의사소통 기회가 뭐야?",
        "대학 입학 단계에서의 의사소통 기회는 대학과 연결하고...",
        "그럼 이런 의사소통 기회를 잘 활용하려면 어떻게 해야 해?"
    ]
    prompt = " ".join(history).strip()
    ids = torch.tensor([sp.EncodeAsIds(prompt)], device=DEVICE)
    if BEAM_WIDTH > 1:
        gen = beam_search(ids)
    else:
        gen = sample_sequence(ids)

    out_ids = gen[0, ids.size(1):].tolist()
    print("=== OUTPUT ===")
    print(sp.DecodeIds(out_ids))
