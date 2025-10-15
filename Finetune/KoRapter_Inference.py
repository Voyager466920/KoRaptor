import torch
import sentencepiece as spm
from collections import Counter
from Finetune.LatentMoE import LatentMoE

checkpoint_path = r"C:\junha\Git\BFG_2B\Checkpoints\A_HuggingFace_KoRapter150M_Kowiki_AIHub_lr_1e_3\model_epoch_5.pt"
tokenizer_model_path = r"C:\junha\Git\BFG_2B\Tokenizer\spm_kowiki.model"
prompt = "대통령은"

MAX_SEQ_LEN = 256
EMBED_DIM = 640
LATENT_DIM = 160
MLP_DIM = 1536
NUM_LAYERS = 8
NUM_HEADS = 8
DROPOUT = 0.1
NUM_EXPERTS = 6
EXPERTS_PER_TOKEN = 2
BALANCE_LOSS_WEIGHT = 0.01

max_new_tokens = 100
temperature = 0.9
top_k = 10
top_p = 0.9
repetition_penalty = 1.2
no_repeat_ngram = 3
NEG_INF = -1e9

def subsequent_mask(seq_len, device=None):
    return torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1).unsqueeze(0).unsqueeze(1)

def load_model(cp_path, tk_path, device):
    sp = spm.SentencePieceProcessor()
    sp.Load(tk_path)
    vocab_size = sp.GetPieceSize()
    model = LatentMoE(vocab_size=vocab_size, max_seq_len=MAX_SEQ_LEN, embed_dim=EMBED_DIM, latent_dim=LATENT_DIM, mlp_dim=MLP_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT, num_heads=NUM_HEADS, num_experts=NUM_EXPERTS, experts_per_token=EXPERTS_PER_TOKEN, balance_loss_weight=BALANCE_LOSS_WEIGHT).to(device)
    state = torch.load(cp_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, sp

def enforce_no_repeat_ngram(logits, generated, n):
    if n <= 1 or generated.size(1) < n:
        return logits
    prev = tuple(generated[0, -(n-1):].tolist())
    banned = set()
    for i in range(generated.size(1) - (n-1)):
        if tuple(generated[0, i:i+n-1].tolist()) == prev:
            banned.add(generated[0, i+n-1].item())
    if banned:
        logits[:, list(banned)] = NEG_INF
    return logits

@torch.no_grad()
def sample_generate(model, sp, device):
    eos_id = sp.eos_id()
    if eos_id < 0:
        try:
            eos_id = sp.PieceToId("</s>")
        except:
            eos_id = None
    prompt_ids = sp.EncodeAsIds(prompt)
    prompt_ids = prompt_ids[-(MAX_SEQ_LEN - 1):]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        cur = ids[:, -MAX_SEQ_LEN:]
        out = model(cur)
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1].unsqueeze(-1)] = NEG_INF
        if top_p and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            mask = cum > top_p
            if mask.any():
                cutoff = torch.argmax(mask.int(), dim=-1).item()
                sorted_logits[:, cutoff+1:] = NEG_INF
            logits = torch.scatter(torch.full_like(logits, NEG_INF), dim=-1, index=sorted_idx, src=sorted_logits)
        from collections import Counter
        counts = Counter(ids.view(-1).tolist())
        for tid, c in counts.items():
            if c > 1:
                logits[:, int(tid)] /= (repetition_penalty ** (c - 1))
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.argmax(logits, dim=-1, keepdim=True) if (not torch.isfinite(probs).all() or probs.sum() <= 0) else torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_token], dim=1)
        if eos_id is not None and int(next_token.item()) == int(eos_id):
            break
    gen = ids[0].tolist()[len(prompt_ids):]
    return sp.DecodeIds(gen)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sp = load_model(checkpoint_path, tokenizer_model_path, device)
    out = sample_generate(model, sp, device)
    print(out)
