import os
from datasets import load_dataset, Dataset
import sentencepiece as spm

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(r"C:\junha\Git\BFG_2B\Tokenizer\spm_bc.model")

ds_stream = load_dataset("Geralt-Targaryen/openwebtext2", split="train", streaming=True)

subset = []
total_tokens = 0
TOKEN_LIMIT = 200_000_000

for sample in ds_stream:
    ids = tokenizer.EncodeAsIds(sample["text"])
    total_tokens += len(ids)
    subset.append(sample)
    if total_tokens >= TOKEN_LIMIT:
        break

new_ds = Dataset.from_list(subset)
splits = new_ds.train_test_split(test_size=0.1, seed=42)
os.makedirs(r"C:\junha\Datasets\OpenWebText2_200M\train", exist_ok=True)
os.makedirs(r"C:\junha\Datasets\OpenWebText2_200M\val", exist_ok=True)
splits["train"].save_to_disk(r"C:\junha\Datasets\OpenWebText2_200M\train")
splits["test"].save_to_disk(r"C:\junha\Datasets\OpenWebText2_200M\val")

print("OpenWebText2 200M 토큰 서브셋 저장 완료")
