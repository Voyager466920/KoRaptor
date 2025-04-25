import torch
from torch.utils.data import Dataset


class Datasets(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len=256, stride=None):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_len = max_seq_len
        self.stride = stride if stride is not None else max_seq_len // 2
        self.samples_map = []
        with open(file_path, 'rb') as f:
            while True:
                pos = f.tell()
                raw = f.readline()
                if not raw:
                    break
                try:
                    line = raw.decode('utf-8').strip()
                except UnicodeDecodeError:
                    line = raw.decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                ids = self.tokenizer.EncodeAsIds(line)
                for start in range(0, len(ids), self.stride):
                    chunk = ids[start:start + self.max_len]
                    if len(chunk) < 2:
                        break
                    self.samples_map.append((pos, start))

    def __len__(self):
        return len(self.samples_map)

    def __getitem__(self, idx):
        pos, start = self.samples_map[idx]
        with open(self.file_path, 'rb') as f:
            f.seek(pos)
            raw = f.readline()
        try:
            line = raw.decode('utf-8').strip()
        except UnicodeDecodeError:
            line = raw.decode('utf-8', errors='ignore').strip()
        ids = self.tokenizer.EncodeAsIds(line)
        chunk = ids[start:start + self.max_len]
        if len(chunk) > self.max_len:
            chunk = chunk[:self.max_len]
        input_ids = chunk[:-1]
        labels = chunk[1:]
        pad_len = self.max_len - len(chunk)
        input_ids = input_ids + [0] * pad_len
        labels = labels + [-100] * pad_len
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long)
        )
