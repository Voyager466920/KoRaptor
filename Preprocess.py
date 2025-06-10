import os
from datasets import load_from_disk, interleave_datasets
import sentencepiece as spm

TOKENIZER_MODEL = r"C:\junha\Git\BFG_2B\Tokenizer\spm_bc.model"
BOOK_TRAIN_PATH = r"C:\junha\Datasets\BookCorpus\train"
BOOK_VAL_PATH   = r"C:\junha\Datasets\BookCorpus\val"
WIKI_TRAIN_PATH = r"C:\junha\Datasets\WikiText103\train"
WIKI_VAL_PATH   = r"C:\junha\Datasets\WikiText103\val"
OUTPUT_DIR      = r"C:\junha\Datasets\Preprocessed"
MAX_SEQ_LEN     = 512
STRIDE          = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(TOKENIZER_MODEL)
pad_id = tokenizer.pad_id()
bos_id = tokenizer.bos_id()
eos_id = tokenizer.eos_id()

def chunk_and_tokenize_batch(batch, max_seq_len=MAX_SEQ_LEN, stride=STRIDE):
    all_input_ids = []
    all_labels    = []
    for text in batch["text"]:
        ids = [bos_id] + tokenizer.EncodeAsIds(text) + [eos_id]
        L = len(ids)
        if L <= max_seq_len:
            inp = ids[:-1] + [pad_id] * (max_seq_len - L + 1)
            lbl = ids[1:]  + [pad_id] * (max_seq_len - L + 1)
            all_input_ids.append(inp)
            all_labels.append(lbl)
        else:
            for start in range(0, L - max_seq_len + 1, stride):
                window = ids[start:start + max_seq_len]
                inp = window[:-1]
                lbl = window[1:]
                if len(inp) < max_seq_len:
                    pad_len = max_seq_len - len(inp)
                    inp += [pad_id] * pad_len
                    lbl += [pad_id] * pad_len
                all_input_ids.append(inp)
                all_labels.append(lbl)
    return {"input_ids": all_input_ids, "labels": all_labels}

def preprocess_split(split_paths, out_name):
    dsets = [load_from_disk(p) for p in split_paths]
    ds = interleave_datasets(dsets)
    ds = ds.map(
        chunk_and_tokenize_batch,
        remove_columns=["text"],
        batched=True,
        batch_size=1000,
        num_proc=os.cpu_count()
    )
    ds = ds.filter(lambda ex: len(ex["input_ids"]) > 0)
    save_path = os.path.join(OUTPUT_DIR, out_name)
    ds.save_to_disk(save_path)
    print(f"Saved preprocessed dataset to {save_path}")

if __name__ == "__main__":
    preprocess_split(
        [BOOK_TRAIN_PATH, WIKI_TRAIN_PATH],
        out_name="train_interleaved"
    )
    preprocess_split(
        [BOOK_VAL_PATH, WIKI_VAL_PATH],
        out_name="val_interleaved"
    )
