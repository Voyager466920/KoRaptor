import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sentencepiece as spm
from peft import LoraConfig, get_peft_model, TaskType
from tqdm.auto import tqdm

from Finetune.KorQuAD_Dataset import KorQuADDataset
from Finetune.LatentMoE import LatentMoE, LatentMoEShim
from Finetune.TestStep import test_step
from Finetune.TrainStep import train_step


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

def greedy_generate(model, tok, prompt_ids, max_new_tokens, eos_id, dev, max_seq_len):
    prompt_ids = prompt_ids[-(max_seq_len - 1):]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            cur = ids[:, -max_seq_len:]
            o = model(cur)
            l = o[0] if isinstance(o, tuple) else o
            nxt = l[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            if int(nxt.item()) == int(eos_id):
                break
    gen = ids[0].tolist()[len(prompt_ids):]
    return gen

def eval_em_f1(model, ds, tok, dev, max_new_tokens=64):
    if hasattr(tok, "eos_id") and tok.eos_id() != -1:
        eos_id = tok.eos_id()
    else:
        eos_id = tok.PieceToId("</s>")
    max_len = getattr(ds, "max_seq_len", 256)
    em_sum, f1_sum = 0.0, 0.0
    n = len(ds)
    for i in range(n):
        ex = ds.samples[i]
        p_ids = list(tok.EncodeAsIds(ex["prompt"]))
        gen = greedy_generate(model, tok, p_ids, max_new_tokens, eos_id, dev, max_len)
        pred = tok.DecodeIds(gen)
        golds = [ex["answer"]]
        em, f1 = em_f1_str(pred, golds)
        em_sum += em
        f1_sum += f1
    return em_sum / max(1, n), f1_sum / max(1, n)


def main():
    set_seed(42)
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    MAX_SEQ_LEN = 256
    BATCH_SIZE = 150
    LR = 1e-3
    EPOCHS = 10
    ACCUM = 1
    TOKENIZER_PATH = r"C:\junha\Git\BFG_2B\Tokenizer\spm_kowiki.model"
    K1_TRAIN = r"C:\junha\Datasets\KoRaptor_FineTuning\KorQuAD_1_0\train"
    K1_DEV   = r"C:\junha\Datasets\KoRaptor_FineTuning\KorQuAD_1_0\val"
    PRETRAIN = r"C:\junha\Git\BFG_2B\Checkpoints\A_HuggingFace_KoRapter150M_Kowiki_AIHub_lr_1e_3\model_epoch_4.pt"
    SAVE_DIR = r"C:\junha\Git\BFG_2B\Checkpoints\korquad_lora"
    os.makedirs(SAVE_DIR, exist_ok=True)
    tok = spm.SentencePieceProcessor()
    tok.Load(TOKENIZER_PATH)
    try:
        pad_id = tok.pad_id()
        if pad_id == -1:
            pad_id = tok.PieceToId("<pad>")
    except:
        pad_id = tok.PieceToId("<pad>")
    tr = KorQuADDataset([K1_TRAIN], tok, MAX_SEQ_LEN)
    va = KorQuADDataset([K1_DEV], tok, MAX_SEQ_LEN)
    trl = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_builder(pad_id))
    val = DataLoader(va, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_builder(pad_id))
    base = LatentMoE(vocab_size=tok.GetPieceSize(), max_seq_len=MAX_SEQ_LEN, embed_dim=640, latent_dim=160,
                     mlp_dim=1536, num_layers=8, num_heads=8, dropout=0.1, num_experts=6,
                     experts_per_token=2, balance_loss_weight=0.01)
    base.load_state_dict(torch.load(PRETRAIN, map_location="cpu"))
    base.to(dev)
    shim = LatentMoEShim(base).to(dev)
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8,
                     lora_alpha=16, lora_dropout=0.1,
                     target_modules=["q_proj","dkv_proj","up_proj_k","up_proj_v","out_proj","fc1","fc2"])
    model = get_peft_model(shim, cfg).to(dev)
    opt = optim.AdamW(model.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.1)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    for e in tqdm(range(1, EPOCHS+1)):
        trp = train_step(model, trl, loss_fn, opt, dev, ACCUM, True)
        vpp, vac = test_step(model, val, loss_fn, dev, True)
        val_em, val_f1 = eval_em_f1(model, va, tok, dev, max_new_tokens=64)
        sch.step()
        print(f"[Epoch {e}] train_ppl={trp:.2f}, val_ppl={vpp:.2f}, val_acc={vac*100:.2f}%, val_EM={val_em*100:.2f}%, val_F1={val_f1*100:.2f}%")
        d = os.path.join(SAVE_DIR, f"epoch_{e}")
        os.makedirs(d, exist_ok=True)
        model.save_pretrained(d)

if __name__ == "__main__":
    main()
