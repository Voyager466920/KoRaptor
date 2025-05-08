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
        if self._use_special_kw:                     # HF tokenizer
            return self.tok.encode(text, add_special_tokens=False)
        else:                                        # SentencePiece
            return self.tok.encode(text)             # or EncodeAsIds(text)

    def _token_chunks(self):
        for row in self.hf_iter:
            ids = self._ids(row["text"]) #row 안에sentence나 text
            for i in range(0, len(ids) - self.stride, self.stride):
                chunk = ids[i : i + self.max_seq_len + 1]
                if len(chunk) < self.max_seq_len + 1:
                    break
                yield chunk

    def __iter__(self):
        for chunk in self._token_chunks():
            inp  = torch.tensor(chunk[:-1], dtype=torch.long)
            targ = torch.tensor(chunk[1:],  dtype=torch.long)
            yield inp, targ