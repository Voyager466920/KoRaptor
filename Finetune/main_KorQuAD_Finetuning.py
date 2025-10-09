import os, json, math, random
from glob import glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from peft import LoraConfig, get_peft_model, TaskType
from tqdm.auto import tqdm
from datasets import load_from_disk
from Finetune.LatentMoE import LatentMoE, LatentMoEShim


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class KorQuADDataset(Dataset):
    def __init__(self, sources, tokenizer, max_seq_len=256, eos_fallback="</s>"):
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        try:
            self.pad_id = self.tok.pad_id()
            if self.pad_id == -1:
                self.pad_id = self.tok.PieceToId("<pad>")
        except:
            self.pad_id = self.tok.PieceToId("<pad>")
        try:
            self.eos_id = self.tok.eos_id()
            if self.eos_id == -1:
                self.eos_id = self.tok.PieceToId("</s>")
        except:
            self.eos_id = self.tok.PieceToId(eos_fallback)
        if not isinstance(sources, (list, tuple)):
            sources = [sources]
        self.samples = []
        for s in sources:
            self._ingest(s)
        if not self.samples:
            raise RuntimeError("No samples loaded")

    def _emit(self, q, c, a):
        q, c, a = (q or "").strip(), (c or "").strip(), (a or "").strip()
        if q and c and a:
            self.samples.append({"prompt": f"질문: {q}\n지문: {c}\n정답: ", "answer": a})

    def _ans_text(self, answers):
        if isinstance(answers, dict) and "text" in answers and isinstance(answers["text"], list) and answers["text"]:
            return str(answers["text"][0])
        return ""

    def _is_arrow_dir(self, p):
        d = Path(p)
        if not d.is_dir():
            return False
        if (d / "dataset_info.json").exists():
            return True
        if any(d.glob("*.arrow")) or any((d / "data").glob("*.arrow")):
            return True
        return False

    def _ingest(self, src):
        if self._is_arrow_dir(src):
            ds = load_from_disk(src)
            for ex in ds:
                q = ex.get("question", "")
                c = ex.get("context", "")
                a = self._ans_text(ex.get("answers", {}))
                self._emit(q, c, a)
            return
        paths = []
        if Path(src).is_dir():
            paths = list(Path(src).glob("**/*.*"))
            paths = [str(p) for p in paths if p.suffix.lower() in {".json", ".jsonl"}]
        else:
            if any(ch in src for ch in "*?"):
                paths = glob(src)
            elif Path(src).exists():
                paths = [src]
        for fp in paths:
            with open(fp, "r", encoding="utf-8") as f:
                head = f.read(2048)
                f.seek(0)
                is_jsonl = ("\n" in head) and not head.strip().startswith("{")
                if is_jsonl:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ex = json.loads(line)
                        except:
                            continue
                        q = ex.get("question", "")
                        c = ex.get("context", "")
                        a = self._ans_text(ex.get("answers", {}))
                        self._emit(q, c, a)
                else:
                    obj = json.load(f)
                    if isinstance(obj, dict) and "data" in obj:
                        for art in obj["data"]:
                            for para in art.get("paragraphs", []):
                                c = para.get("context", "")
                                for qa in para.get("qas", []):
                                    q = qa.get("question", "")
                                    a = self._ans_text(qa.get("answers", {}))
                                    self._emit(q, c, a)
                    elif isinstance(obj, dict):
                        q = obj.get("question", "")
                        c = obj.get("context", "")
                        a = self._ans_text(obj.get("answers", {}))
                        self._emit(q, c, a)
                    elif isinstance(obj, list):
                        for ex in obj:
                            q = ex.get("question", "")
                            c = ex.get("context", "")
                            a = self._ans_text(ex.get("answers", {}))
                            self._emit(q, c, a)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ex = self.samples[i]
        p = list(self.tok.EncodeAsIds(ex["prompt"]))
        a = list(self.tok.EncodeAsIds(ex["answer"])) + [self.eos_id]
        total = len(p) + len(a)
        if total > self.max_seq_len:
            over = total - self.max_seq_len
            if over < len(p):
                p = p[over:]
            else:
                cut = over - len(p)
                p = []
                a = a[: max(1, len(a) - cut)]
        x = p + a
        y = [-100] * len(p) + a[:]
        x = x[: self.max_seq_len]
        y = y[: self.max_seq_len]
        return {"input_ids": torch.tensor(x, dtype=torch.long), "labels": torch.tensor(y, dtype=torch.long)}

def collate_fn_builder(pad_id):
    def _fn(b):
        xs = [x["input_ids"] for x in b]
        ys = [x["labels"] for x in b]
        xs = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=pad_id)
        ys = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
        return {"input_ids": xs, "labels": ys}
    return _fn

def train_step(model, loader, loss_fn, opt, dev, acc_steps=1, amp=True):
    model.train()
    sc = torch.amp.GradScaler(enabled=amp)
    opt.zero_grad(set_to_none=True)
    ce_sum, tok_sum = 0.0, 0
    for i, b in enumerate(loader, 1):
        x = b["input_ids"].to(dev)
        y = b["labels"].to(dev)
        with torch.amp.autocast(device_type="cuda", enabled=(amp and dev.type == "cuda")):
            o = model(x)
            l = o[0] if isinstance(o, tuple) else o
            ce = loss_fn(l.view(-1, l.size(-1)), y.view(-1))
            loss = ce / acc_steps
        sc.scale(loss).backward()
        if i % acc_steps == 0:
            sc.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            sc.step(opt)
            sc.update()
            opt.zero_grad(set_to_none=True)
        m = (y.view(-1) != -100)
        ce_sum += ce.item() * m.sum().item()
        tok_sum += m.sum().item()
    return math.exp(ce_sum / max(1, tok_sum))

def test_step(model, loader, loss_fn, dev, amp=True):
    model.eval()
    ce_sum, tok_sum, cor = 0.0, 0, 0
    with torch.no_grad():
        for b in loader:
            x = b["input_ids"].to(dev)
            y = b["labels"].to(dev)
            with torch.amp.autocast(device_type="cuda", enabled=(amp and dev.type == "cuda")):
                o = model(x)
                l = o[0] if isinstance(o, tuple) else o
                ce = loss_fn(l.view(-1, l.size(-1)), y.view(-1))
            m = (y.view(-1) != -100)
            ce_sum += ce.item() * m.sum().item()
            tok_sum += m.sum().item()
            p = l.argmax(dim=-1)
            cor += (p.view(-1)[m] == y.view(-1)[m]).sum().item()
    return math.exp(ce_sum / max(1, tok_sum)), cor / max(1, tok_sum)

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
