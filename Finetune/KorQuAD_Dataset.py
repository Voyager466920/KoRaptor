import json
from glob import glob
from pathlib import Path

import torch
from torch.utils.data import Dataset
from datasets import load_from_disk

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
