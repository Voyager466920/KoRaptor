from torch.utils.data import IterableDataset
import torch


class StreamingMultipleDatset(IterableDataset):
    def __init__(self, splits, tokenizer, max_seq_len: int, stride: int = None):
        self.iterables = splits
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride or max_seq_len
        self.pad_id = tokenizer.pad_id()
        self.bos_id = tokenizer.bos_id()
        self.eos_id = tokenizer.eos_id()

    def __iter__(self):
        for split in self.iterables:
            for ex in split:
                text = ex.get("text")
                if not text or not isinstance(text, str):
                    continue

                ids = self.tokenizer.EncodeAsIds(text)
                if len(ids) < 1:
                    continue

                # add BOS/EOS
                ids = [self.bos_id] + ids + [self.eos_id]
                L = len(ids)

                # short sequence: pad to full length
                if L <= self.max_seq_len:
                    pad_len = self.max_seq_len - L + 1
                    input_ids = ids[:-1] + [self.pad_id] * pad_len
                    labels = ids[1:] + [self.pad_id] * pad_len
                    yield {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long),
                    }
                else:
                    for start in range(0, L - self.max_seq_len + 1, self.stride):
                        window = ids[start:start + self.max_seq_len]
                        input_ids = window[:-1]
                        labels = window[1:]
                        pad_len = self.max_seq_len - len(input_ids)
                        if pad_len > 0:
                            input_ids += [self.pad_id] * pad_len
                            labels += [self.pad_id] * pad_len

                        yield {
                            "input_ids": torch.tensor(input_ids, dtype=torch.long),
                            "labels": torch.tensor(labels, dtype=torch.long),
                        }
