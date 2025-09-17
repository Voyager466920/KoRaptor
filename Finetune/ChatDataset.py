import os
import json
import torch
from torch.utils.data import Dataset

class ChatDataset(Dataset):
    def __init__(self, path_jsonl: str, tokenizer, max_seq_len: int):
        self.examples = []
        eos_id = tokenizer.eos_id() if hasattr(tokenizer, "eos_id") else None
        if eos_id is None or eos_id < 0:
            raise ValueError("Tokenizer must define a valid EOS token.")

        def add_example(prompt: str, resp: str):
            if not isinstance(prompt, str) or not isinstance(resp, str):
                return
            prompt = prompt.strip()
            resp = resp.strip()
            if not prompt or not resp:
                return

            ids_prompt = list(tokenizer.EncodeAsIds(prompt))
            ids_resp = list(tokenizer.EncodeAsIds(resp))

            full = ids_prompt + [eos_id] + ids_resp + [eos_id]
            len_prompt_with_eos = len(ids_prompt) + 1

            if len(full) > max_seq_len:
                overflow = len(full) - max_seq_len
                full = full[overflow:]
                remain_prompt = max(0, len_prompt_with_eos - overflow)
            else:
                remain_prompt = len_prompt_with_eos

            if len(full) < 2:
                return

            input_ids = torch.tensor(full[:-1], dtype=torch.long)
            labels = torch.tensor(full[1:], dtype=torch.long)

            mask_len = min(remain_prompt, labels.size(0))
            if mask_len > 0:
                labels[:mask_len] = -100

            self.examples.append({"input_ids": input_ids, "labels": labels})

        with open(path_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "prompt" in obj and "response" in obj:
                    add_example(obj["prompt"], obj["response"])
                    continue

                convs = obj.get("conversations") or []
                if isinstance(convs, list):
                    pending = None
                    for turn in convs:
                        role = turn.get("from")
                        text = turn.get("value")
                        if role == "human":
                            pending = text
                        elif role == "gpt" and pending is not None:
                            add_example(pending, text)
                            pending = None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):
            return [self._get_single(i) for i in idx]
        return self._get_single(idx)

    def _get_single(self, i):
        return self.examples[i]
