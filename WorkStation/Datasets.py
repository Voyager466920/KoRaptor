import torch
from torch.utils.data import Dataset

class Datasets(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len=256):
        self.tokenizer = tokenizer
        self.max_len   = max_seq_len
        self.samples   = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = self.tokenizer.EncodeAsIds(line)
                if len(ids) > self.max_len:
                    ids = ids[:self.max_len]
                input_ids = ids[:-1]
                labels    = ids[1:]
                pad_len   = self.max_len - len(ids)
                input_ids = input_ids + [0] * pad_len
                labels    = labels    + [-100] * pad_len
                self.samples.append((
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(labels,    dtype=torch.long)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
