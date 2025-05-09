import torch
from torch.utils.data import IterableDataset
import inspect

class HFStreamingDataset(IterableDataset):
    def __init__(self, hf_iter, tokenizer, max_seq_len=4096, stride=128):
        super().__init__()
        self.hf_iter = hf_iter
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride

        sig = inspect.signature(self.tok.encode)
        self._use_special_kw = "add_special_tokens" in sig.parameters

    def _ids(self, text: str):
        if self._use_special_kw:  # HF tokenizer (e.g., gpt2)
            ids = self.tok.encode(
                text,
                add_special_tokens=False,
                max_length=self.max_seq_len,
                truncation=True
            )
        else:  # SentencePiece
            ids = self.tok.encode(text)
            ids = ids[:self.max_seq_len]  # Truncate to max_seq_len
        if len(ids) > self.max_seq_len:
            print(f"Warning: Truncated sequence of length {len(ids)} to {self.max_seq_len}")
        return ids

    def _token_chunks(self):
        for row in self.hf_iter:
            text = row["text"]
            if not text or not isinstance(text, str):
                continue
            ids = self._ids(text)
            if len(ids) < 2:  # Skip sequences too short
                continue
            for i in range(0, len(ids) - self.stride, self.stride):
                chunk = ids[i: i + self.max_seq_len + 1]
                if len(chunk) != self.max_seq_len + 1:  # Ensure exact length
                    break
                yield chunk

    def __iter__(self):
        for chunk in self._token_chunks():
            inp = torch.tensor(chunk[:-1], dtype=torch.long)
            targ = torch.tensor(chunk[1:], dtype=torch.long)
            yield inp, targ