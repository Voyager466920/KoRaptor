import torch
import sentencepiece as spm
from Models.LatentMoE import LatentMoE


#checkpoint_path = r"C:\junha\Git\BFG_2B\Checkpoints\Rapter72M_Wiki_Book\72M_model_epoch_10.pt"
checkpoint_path = r"C:\junha\Git\BFG_2B\Checkpoints\Rapter150M_Wiki_Book_Web_406M\72M_309MD_model_epoch_10.pt"
tokenizer_model_path = r"C:\junha\Git\BFG_2B\Tokenizer\spm_bc.model"
prompt = "Who are you"
max_length = 100
temperature = 1.0
top_k = 10
top_p = 0.9
beam_width = 3

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
    state = torch.load(cp_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, sp

@torch.no_grad()
def sample_sequence(model, sp, device, max_length, temperature, top_k, top_p):
    eos_id = sp.eos_id()
    input_ids = torch.tensor([sp.EncodeAsIds(prompt)], dtype=torch.long, device=device)
    generated = input_ids

    for _ in range(max_length):
        out = model(generated)
        logits = out.logits if hasattr(out, 'logits') else (out[0] if isinstance(out, tuple) else out)
        logits = logits[:, -1, :] / temperature

        # top-k
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1].unsqueeze(-1)] = -float('Inf')
        # top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            cutoff = cumulative_probs > top_p
            if cutoff.any():
                cutoff_index = torch.nonzero(cutoff)[0,1]
                mask = sorted_logits < sorted_logits[:, cutoff_index].unsqueeze(-1)
                logits[mask.scatter(1, sorted_indices, mask)] = -float('Inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
        if next_token.item() == eos_id:
            break

    return sp.DecodeIds(generated.squeeze().tolist())

@torch.no_grad()
def beam_search(model, sp, device, max_length, temperature, beam_width):
    eos_id = sp.eos_id()
    initial = torch.tensor([sp.EncodeAsIds(prompt)], dtype=torch.long, device=device)
    beams = [(initial, 0.0)]  # (sequence, cumulative log-prob)

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
                candidates.append((new_seq, score + lp))
        # prune
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        # if all end with eos, break
        if all(seq[0, -1].item() == eos_id for seq, _ in beams):
            break

    best_seq = beams[0][0]
    return sp.DecodeIds(best_seq.squeeze().tolist())

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, sp = load_model(checkpoint_path, tokenizer_model_path, device)
    if beam_width > 1:
        output = beam_search(model, sp, device, max_length, temperature, beam_width)
    else:
        output = sample_sequence(model, sp, device, max_length, temperature, top_k, top_p)
    print("\n=== Generated Output ===\n")
    print(output)
