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
from Finetune.LatentMoE import LatentMoE, LatentMoEShim
from Finetune.Test_Step import test_step
from Finetune.Train_Step import train_step

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def collate_fn_builder(pad_id, max_len):
    def _fn(b):
        xs = [x["input_ids"] for x in b]
        ys = [x["labels"] for x in b]
        xs = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=pad_id)
        ys = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
        if xs.size(1) < max_len:
            pad_w = max_len - xs.size(1)
            xs = torch.cat([xs, torch.full((xs.size(0), pad_w), pad_id, dtype=xs.dtype)], dim=1)
            ys = torch.cat([ys, torch.full((ys.size(0), pad_w), -100, dtype=ys.dtype)], dim=1)
        elif xs.size(1) > max_len:
            xs = xs[:, :max_len]
            ys = ys[:, :max_len]
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

DEC_TEMPERATURE = 0.8
DEC_TOP_K = 10
DEC_TOP_P = 0.9
DEC_REP_PEN = 1.2
DEC_NGRAM = 3
DEC_MAX_NEW = 8
NEG_INF = -1e9

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
def beam_search_decode(model, tok, prompt_ids, eos_id, pad_id, dev, max_seq_len,
                       beam_width=1, alpha=0.7, max_new_tokens=8, temperature=0.0):
    prompt_ids = prompt_ids[-max_seq_len:]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
    beams = [(input_ids, 0.0)]
    for _ in range(max_new_tokens):
        candidates = []
        for seq, score in beams:
            cur = seq[:, -max_seq_len:]
            out = model(cur)
            logits = out[0] if isinstance(out, tuple) else out
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
            for lp, idx in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                new_seq = torch.cat((seq, torch.tensor([[idx]], device=dev)), dim=1)
                new_score = score + lp / (new_seq.size(1) ** alpha)
                candidates.append((new_seq, new_score))
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(seq[0, -1].item() == eos_id for seq, _ in beams):
            break
    best_seq = beams[0][0]
    return best_seq[0].tolist()[len(prompt_ids):]

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
        gen = beam_search_decode(model, tok, p_ids, eos_id, pad_id, dev, max_len,
                                 beam_width=3, alpha=0.7, max_new_tokens=max_new_tokens)
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

    TOKENIZER_PATH = r"C:\junha\Git\BFG_2B\Tokenizer\spm_koraptor.model"
    KorQuADV1_Train = r"C:\junha\Datasets\KoRaptor_FineTuning\KorQuAD_1_0\train"
    KorQuADV1_Val   = r"C:\junha\Datasets\KoRaptor_FineTuning\KorQuAD_1_0\val"
    Pretrained_Model = r"C:\junha\Git\BFG_2B\Checkpoints\KoRapter150M_Kowiki_251004\model_epoch_4.pt"
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
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_fn_builder(pad_id, MAX_SEQ_LEN))
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate_fn_builder(pad_id, MAX_SEQ_LEN))

    base_model = LatentMoE(vocab_size=tokenizer.GetPieceSize(), max_seq_len=MAX_SEQ_LEN, embed_dim=640, latent_dim=160, mlp_dim=1536, num_layers=8, num_heads=8, dropout=0.1, num_experts=6, experts_per_token=2, balance_loss_weight=0.01)
    base_model.load_state_dict(torch.load(Pretrained_Model, map_location="cpu"))
    base_model.to(device)

    shim = LatentMoEShim(base_model).to(device)
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1,
                     target_modules=["q_proj","dkv_proj","up_proj_k","up_proj_v","out_proj","fc1","fc2"])
    peft_model = get_peft_model(shim, cfg).to(device)

    model = peft_model.to(device)

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
