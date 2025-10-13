import os, random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
import sentencepiece as spm
from peft import LoraConfig, get_peft_model, TaskType
from tqdm.auto import tqdm
from collections import Counter

from Finetune.KorQuAD_Dataset import KorQuADDataset
from Finetune.LatentMoE import LatentMoE, LatentMoEShim, subsequent_mask
from Finetune.Test_Step import test_step
from Finetune.Train_Step import train_step

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def collate_fn_builder(pad_id):
    def _fn(b):
        xs = [x["input_ids"] for x in b]
        ys = [x["labels"] for x in b]
        xs = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=pad_id)
        ys = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
        return {"input_ids": xs, "labels": ys}
    return _fn

def _norm(s):
    return "".join((s or "").split())

def em_f1_str(pred, golds):
    p = _norm(pred)
    em = 1.0 if any(p == _norm(g) for g in golds) else 0.0
    best_f1 = 0.0
    for g in golds:
        a = list(p); b = list(_norm(g))
        ca = {}; cb = {}
        for ch in a: ca[ch] = ca.get(ch, 0) + 1
        for ch in b: cb[ch] = cb.get(ch, 0) + 1
        overlap = sum(min(ca.get(k,0), cb.get(k,0)) for k in set(ca) | set(cb))
        if len(a) == 0 or len(b) == 0:
            f1 = 1.0 if len(a) == len(b) else 0.0
        else:
            prec = overlap / len(a); rec = overlap / len(b)
            f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        if f1 > best_f1: best_f1 = f1
    return em, best_f1

# ===== 디코딩 하이퍼(EM/F1에 유리하게 짧고 보수적으로) =====
DEC_TEMPERATURE = 0.7
DEC_TOP_K = 20
DEC_TOP_P = 0.9
DEC_REP_PEN = 1.2
DEC_NGRAM = 3
DEC_MAX_NEW = 8
NEG_INF = -1e9

# ===== 모델에 attn_mask를 항상 주입하는 래퍼 =====
class MaskedModel(nn.Module):
    def __init__(self, base_model: nn.Module, pad_id: int):
        super().__init__()
        self.base = base_model
        self.pad_id = pad_id
    def forward(self, input_ids, **kwargs):
        x = input_ids
        pad_m  = (x == self.pad_id).unsqueeze(1).unsqueeze(2)           # [B,1,1,S] (True=mask)
        causal = subsequent_mask(x.size(1), device=x.device)            # [1,1,S,S]
        attn_mask = pad_m | causal                                      # 브로드캐스트 OR
        kwargs["attn_mask"] = attn_mask
        return self.base(input_ids, **kwargs)

def _enforce_no_repeat_ngram(logits, generated, n):
    if generated.size(1) < n:
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
def sample_decode(model, tok, prompt_ids, eos_id, pad_id, dev, max_seq_len,
                  temperature=DEC_TEMPERATURE, top_k=DEC_TOP_K, top_p=DEC_TOP_P,
                  rep_pen=DEC_REP_PEN, no_repeat_ngram=DEC_NGRAM, max_new_tokens=DEC_MAX_NEW):
    prompt_ids = prompt_ids[-(max_seq_len - 1):]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
    # 특수 토큰 블록리스트
    ban_ids = set()
    for name in ("<pad>", "<s>", "</s>", "<unk>"):
        try:
            tid = tok.PieceToId(name)
            if tid >= 0: ban_ids.add(int(tid))
        except: pass
    ban_ids.add(int(pad_id))

    for _ in range(max_new_tokens):
        cur = ids[:, -max_seq_len:]
        if cur.size(1) < max_seq_len:
            pad = torch.full((1, max_seq_len - cur.size(1)), pad_id, dtype=torch.long, device=dev)
            cur = torch.cat([pad, cur], dim=1)

        # 마스크 생성 (좌패딩 + causal)
        pad_m  = (cur == pad_id).unsqueeze(1).unsqueeze(2)          # [B,1,1,S]
        causal = subsequent_mask(cur.size(1), device=cur.device)    # [1,1,S,S]
        attn_mask = pad_m | causal

        out = model(cur, attn_mask=attn_mask)
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        if ban_ids:
            logits[:, list(ban_ids)] = NEG_INF

        if top_k and top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1].unsqueeze(-1)] = NEG_INF
        if top_p and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            cutoff_mask = cumulative_probs > top_p
            if cutoff_mask.any():
                cutoff_pos = torch.argmax(cutoff_mask.int(), dim=-1).item()
                sorted_logits[:, cutoff_pos+1:] = NEG_INF
            logits = torch.scatter(torch.full_like(logits, NEG_INF), dim=-1, index=sorted_idx, src=sorted_logits)

        counts = Counter(ids.view(-1).tolist())
        for tid, c in counts.items():
            if c > 1:
                logits[:, int(tid)] /= (rep_pen ** (c - 1))

        logits = _enforce_no_repeat_ngram(logits, ids, no_repeat_ngram)
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.argmax(logits, dim=-1, keepdim=True) if (not torch.isfinite(probs).all() or probs.sum() <= 0) else torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt.item()) == int(eos_id):
            break
    gen = ids[0].tolist()[len(prompt_ids):]
    return gen

def eval_em_f1(model, ds, tok, dev, pad_id, max_new_tokens=DEC_MAX_NEW, log_every=500, max_logs=10):
    if hasattr(tok, "eos_id") and tok.eos_id() != -1:
        eos_id = tok.eos_id()
    else:
        eos_id = tok.PieceToId("</s>")
    max_len = getattr(ds, "max_seq_len", 256)
    em_sum, f1_sum = 0.0, 0.0
    n = len(ds)
    tp_sum, fp_sum, fn_sum = 0, 0, 0
    logs = 0
    for i in range(n):
        ex = ds.samples[i]
        p_ids = list(tok.EncodeAsIds(ex["prompt"]))
        gen = sample_decode(model, tok, p_ids, eos_id, pad_id, dev, max_len,
                            temperature=DEC_TEMPERATURE, top_k=DEC_TOP_K, top_p=DEC_TOP_P,
                            rep_pen=DEC_REP_PEN, no_repeat_ngram=DEC_NGRAM, max_new_tokens=max_new_tokens)
        pred = tok.DecodeIds(gen)
        golds = [ex["answer"]]
        em, f1 = em_f1_str(pred, golds)
        em_sum += em
        f1_sum += f1
        a = list("".join((pred or "").split()))
        b = list("".join((golds[0] or "").split()))
        ca = {}; cb = {}
        for ch in a: ca[ch] = ca.get(ch, 0) + 1
        for ch in b: cb[ch] = cb.get(ch, 0) + 1
        overlap = sum(min(ca.get(k,0), cb.get(k,0)) for k in set(ca) | set(cb))
        tp = overlap
        fp = max(0, len(a) - overlap)
        fn = max(0, len(b) - overlap)
        tp_sum += tp; fp_sum += fp; fn_sum += fn
        if (i % log_every == 0) and (logs < max_logs):
            q = ex["prompt"].split("\n")[0].replace("질문: ","")
            print("—"*40)
            print(f"idx={i}")
            print(f"Q: {q}")
            print(f"Gold: {golds[0]}")
            print(f"Pred: {pred}")
            print(f"EM={em:.3f}, F1={f1:.3f}")
            logs += 1
    prec = 0.0 if (tp_sum + fp_sum) == 0 else tp_sum / (tp_sum + fp_sum)
    rec = 0.0 if (tp_sum + fn_sum) == 0 else tp_sum / (tp_sum + fn_sum)
    f1_macro = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    print("—"*40)
    print("Confusion Matrix (char-level)")
    print(f"TP={tp_sum}  FP={fp_sum}")
    print(f"FN={fn_sum}  TN=0")
    print(f"Precision={prec:.4f}  Recall={rec:.4f}  F1={f1_macro:.4f}")
    return em_sum / max(1, n), f1_sum / max(1, n)

def main():
    set_seed(42)
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    MAX_SEQ_LEN = 256
    BATCH_SIZE = 150
    LR = 1e-3
    EPOCHS = 10
    ACCUM_STEPS = 1

    TOKENIZER_PATH = r"C:\junha\Git\BFG_2B\Tokenizer\spm_kowiki.model"
    KorQuADV1_Train = r"C:\junha\Datasets\KoRaptor_FineTuning\KorQuAD_1_0\train"
    KorQuADV1_Val   = r"C:\junha\Datasets\KoRaptor_FineTuning\KorQuAD_1_0\val"
    Pretrained_Model = r"C:\junha\Git\BFG_2B\Checkpoints\A_HuggingFace_KoRapter150M_Kowiki_AIHub_lr_1e_3\model_epoch_4.pt"
    SAVE_DIR = r"C:\junha\Git\BFG_2B\Checkpoints\KorQuAD_V1_LoRA"
    os.makedirs(SAVE_DIR, exist_ok=True)

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(TOKENIZER_PATH)
    pad_id = tokenizer.pad_id()
    if pad_id < 0:
        try:
            pad_id = tokenizer.PieceToId("<pad>")
            assert pad_id >= 0
        except:
            pad_id = 0

    train_dataset = KorQuADDataset([KorQuADV1_Train], tokenizer, MAX_SEQ_LEN)
    val_dataset = KorQuADDataset([KorQuADV1_Val], tokenizer, MAX_SEQ_LEN)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_builder(pad_id))
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_builder(pad_id))

    base_model = LatentMoE(vocab_size=tokenizer.GetPieceSize(), max_seq_len=MAX_SEQ_LEN, embed_dim=640, latent_dim=160, mlp_dim=1536, num_layers=8, num_heads=8, dropout=0.1, num_experts=6, experts_per_token=2, balance_loss_weight=0.01)
    base_model.load_state_dict(torch.load(Pretrained_Model, map_location="cpu"))
    base_model.to(device)

    shim = LatentMoEShim(base_model).to(device)
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1,
                     target_modules=["q_proj","dkv_proj","up_proj_k","up_proj_v","out_proj","fc1","fc2"])
    peft_model = get_peft_model(shim, cfg).to(device)

    model = MaskedModel(peft_model, pad_id).to(device)

    optimizer = AdamW(model.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in tqdm(range(1, EPOCHS+1)):
        train_ppl = train_step(model, train_dataloader, loss_fn, optimizer, device, ACCUM_STEPS, True)
        val_ppl, val_acc = test_step(model, val_dataloader, loss_fn, device, True)
        val_em, val_f1 = eval_em_f1(model, val_dataset, tokenizer, device, pad_id, max_new_tokens=DEC_MAX_NEW)
        scheduler.step()
        print(f"[Epoch {epoch}] train_ppl={train_ppl:.2f}, val_ppl={val_ppl:.2f}, val_acc={val_acc*100:.2f}%, val_EM={val_em*100:.2f}%, val_F1={val_f1*100:.2f}%")
        d = os.path.join(SAVE_DIR, f"epoch_{epoch}")
        os.makedirs(d, exist_ok=True)
        peft_model.save_pretrained(d)

if __name__ == "__main__":
    main()
