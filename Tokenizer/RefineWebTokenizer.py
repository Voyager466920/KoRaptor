import os
from itertools import islice
from datasets import load_from_disk, Dataset, DatasetDict
import sentencepiece as spm

WIKI2_TRAIN = r"C:\junha\Datasets\WikiText2\train"
WIKI103_TRAIN = r"C:\junha\Datasets\WikiText103\train"
OWT2_TRAIN = r"C:\junha\Datasets\OpenWebText2\train"
BOOK_TRAIN = r"C:\junha\Datasets\BookCorpus\train"

OUTPUT_SAMPLE_FILE = r"C:\junha\Datasets\tokenizer_corpus\all_sample.txt"
MODEL_PREFIX = r"C:\junha\Git\BFG_2B\Tokenizer\spm_wiki_book_owt"
VOCAB_SIZE = 35000
CHAR_COVERAGE = 0.9995
MODEL_TYPE = "bpe"
TEXT_KEY = "text"

MAX_LINES_WIKI2 = 300_000
MAX_LINES_WIKI103 = 300_000
MAX_LINES_OWT2 = 300_000
MAX_LINES_BOOK = 300_000

def _as_dataset(obj, split=None):
    if isinstance(obj, DatasetDict):
        return obj[split or "train"]
    return obj

def _iter_text_from_arrow(disk_dir, text_key, max_lines, split=None):
    ds = _as_dataset(load_from_disk(disk_dir), split=split)
    for ex in islice(ds, max_lines):
        txt = (ex.get(text_key) or "").replace("\n", " ").strip()
        if txt:
            yield txt

def build_sample_file():
    os.makedirs(os.path.dirname(OUTPUT_SAMPLE_FILE), exist_ok=True)
    total = 0
    with open(OUTPUT_SAMPLE_FILE, "w", encoding="utf-8") as f:
        for t in _iter_text_from_arrow(WIKI2_TRAIN, TEXT_KEY, MAX_LINES_WIKI2):
            f.write(t + "\n"); total += 1
        for t in _iter_text_from_arrow(WIKI103_TRAIN, TEXT_KEY, MAX_LINES_WIKI103):
            f.write(t + "\n"); total += 1
        for t in _iter_text_from_arrow(OWT2_TRAIN, TEXT_KEY, MAX_LINES_OWT2):
            f.write(t + "\n"); total += 1
        for t in _iter_text_from_arrow(BOOK_TRAIN, TEXT_KEY, MAX_LINES_BOOK):
            f.write(t + "\n"); total += 1
    print(f"[DONE] {OUTPUT_SAMPLE_FILE} ({total} lines)")

def train_sentencepiece():
    spm.SentencePieceTrainer.Train(
        input=OUTPUT_SAMPLE_FILE,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        character_coverage=CHAR_COVERAGE,
        model_type=MODEL_TYPE,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        normalization_rule_name="nfkc",
    )
    print(f"[DONE] {MODEL_PREFIX}.model / {MODEL_PREFIX}.vocab")

if __name__ == "__main__":
    build_sample_file()
    train_sentencepiece()
