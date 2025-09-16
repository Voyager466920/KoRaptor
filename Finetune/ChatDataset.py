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

                convs = obj["conversations"]

                for i in range(0, len(convs) - 1, 2):
                    if convs[i]["from"] == "human" and convs[i + 1]["from"] == "gpt":
                        prompt = convs[i]["value"]
                        resp = convs[i + 1]["value"]

                        text = prompt + eos_piece + resp + eos_piece
                        ids = tokenizer.EncodeAsIds(text)
                        if len(ids) > max_seq_len:
                            ids = ids[-max_seq_len:]

                        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
                        labels = torch.tensor(ids[1:], dtype=torch.long)

                        prompt_piece_ids = tokenizer.EncodeAsIds(prompt + eos_piece)
                        prompt_len = min(len(prompt_piece_ids), labels.size(0))
                        labels[:prompt_len] = -100

                        self.examples.append({
                            "input_ids": input_ids,
                            "labels": labels,
                        })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):
            return [self._get_single(i) for i in idx]
        return self._get_single(idx)

    def _get_single(self, i):
        return self.examples[i]
