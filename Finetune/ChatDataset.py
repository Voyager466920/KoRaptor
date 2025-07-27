import os
import json
import torch
from torch.utils.data import Dataset

class ChatDataset(Dataset):
    def __init__(self, path_jsonl: str, tokenizer, max_seq_len: int):
        self.examples = []
        eos_piece = tokenizer.id_to_piece(tokenizer.eos_id())
        with open(path_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                prompt = obj['prompt']
                resp   = obj['response']
                text   = prompt + eos_piece + resp + eos_piece
                ids    = tokenizer.EncodeAsIds(text)
                if len(ids) > max_seq_len:
                    ids = ids[-max_seq_len:]
                self.examples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):
            return [self._get_single(i) for i in idx]
        return self._get_single(idx)

    def _get_single(self, i):
        ids = self.examples[i]
        return {
            "input_ids": ids[:-1],
            "labels":    ids[1:],
        }
