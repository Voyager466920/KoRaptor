import torch
import sentencepiece as spm
from collections import Counter
from Models.LatentMoE import LatentMoE

checkpoint_path = r"C:\junha\Git\BFG_2B\Checkpoints\KoRapter150M_Kowiki_251004\model_epoch_1.pt"
tokenizer_model_path = r"C:\junha\Git\BFG_2B\Tokenizer\spm_koraptor.model"
prompt = "대통령은"
max_length = 100
temperature = 0.8
top_k = 10
top_p = 0.9
repetition_penalty = 1.2
no_repeat_ngram_size = 3
beam_width = 1
alpha = 0.7

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
NEG_INF = -1e9

def load_model(cp_path, tk_path, device,
               max_seq_len=MAX_SEQ_LEN, embed_dim=EMBED_DIM, latent_dim=LATENT_DIM,
               mlp_dim=MLP_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT,
               num_heads=NUM_HEADS, num_experts=NUM_EXPERTS, experts_per_token=EXPERTS_PER_TOKEN,
               balance_loss_weight=BALANCE_LOSS_WEIGHT):
    sp = spm.SentencePieceProcessor()
    sp.Load(tk_path)
    vocab_size = sp.GetPieceSize()
    model = LatentMoE(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_heads=num_heads,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        balance_loss_weight=balance_loss_weight,
    ).to(device)
    state = torch.load(cp_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, sp

def enforce_no_repeat_ngram(logits, generated, n):
    if generated.size(1) < n:
        return logits
    prev_ngram = tuple(generated[0, -(n-1):].tolist())
    banned = set()
    for i in range(generated.size(1) - (n-1)):
        if tuple(generated[0, i:i+n-1].tolist()) == prev_ngram:
            banned.add(generated[0, i+n-1].item())
    if banned:
        logits[:, list(banned)] = NEG_INF
    return logits

@torch.no_grad()
def sample_sequence(model, sp, device, max_length, temperature, top_k, top_p):
    eos_id = sp.eos_id()
    input_ids = torch.tensor([sp.EncodeAsIds(prompt)], dtype=torch.long, device=device)
    generated = input_ids
    for _ in range(max_length):
        out = model(generated)
        logits = out.logits if hasattr(out, 'logits') else (out[0] if isinstance(out, tuple) else out)
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1].unsqueeze(-1)] = NEG_INF
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            cutoff_mask = cumulative_probs > top_p
            if cutoff_mask.any():
                cutoff_pos = torch.argmax(cutoff_mask.int(), dim=-1).item()
                sorted_logits[:, cutoff_pos+1:] = NEG_INF
            logits = torch.scatter(torch.full_like(logits, NEG_INF),
                                   dim=-1,
                                   index=sorted_idx,
                                   src=sorted_logits)
        token_counts = Counter(generated.view(-1).tolist())
        for token_id, count in token_counts.items():
            if count > 1:
                logits[:, token_id] /= (repetition_penalty ** (count - 1))
        logits = enforce_no_repeat_ngram(logits, generated, no_repeat_ngram_size)
        probs = torch.softmax(logits, dim=-1)
        if not torch.isfinite(probs).all() or probs.sum() <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
        if next_token.item() == eos_id:
            break
    full = generated.squeeze().tolist()
    return sp.DecodeIds(full[len(input_ids[0]):])

@torch.no_grad()
def beam_search(model, sp, device, max_length, temperature, beam_width):
    eos_id = sp.eos_id()
    initial = torch.tensor([sp.EncodeAsIds(prompt)], dtype=torch.long, device=device)
    beams = [(initial, 0.0)]
    for _ in range(max_length):
        candidates = []
        for seq, score in beams:
            out = model(seq)
            logits = out.logits if hasattr(out, 'logits') else (out[0] if isinstance(out, tuple) else out)
            logits = logits[:, -1, :] / temperature
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
            for lp, idx in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                new_seq = torch.cat((seq, torch.tensor([[idx]], device=device)), dim=1)
                length = new_seq.size(1)
                lp_adjusted = lp / (length ** alpha)
                new_score = score + lp_adjusted
                candidates.append((new_seq, new_score))
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(seq[0, -1].item() == eos_id for seq, _ in beams):
            break
    best_seq = beams[0][0]
    full = best_seq.squeeze().tolist()
    return sp.DecodeIds(full[len(initial[0]):])

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, sp = load_model(checkpoint_path, tokenizer_model_path, device)
    if beam_width > 1:
        output = beam_search(model, sp, device, max_length, temperature, beam_width)
    else:
        output = sample_sequence(model, sp, device, max_length, temperature, top_k, top_p)
    print(output)
