import math
import sentencepiece as spm
import torch
from Pretrain.Datasets import Datasets

# 설정
FILE_PATH   = r"C:\junha\Git\BFG_2B\Datasets\train.txt"
MODEL_PATH  = r"C:\junha\Git\BFG_2B\Tokenizer\spm_kowiki.model"
MAX_SEQ_LEN = 512
STRIDE      = 256

# 토크나이저 초기화
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(MODEL_PATH)

# Datasets 인스턴스 생성
dataset = Datasets(FILE_PATH, tokenizer, max_seq_len=MAX_SEQ_LEN, stride=STRIDE)
print(f"Total chunks in dataset: {len(dataset)}")

# Inspect first 5 chunks
good = []
print("Inspecting first 10 chunks:")
for idx in range(10):
    inp, lbl = dataset[idx]
    nonpad_len = (inp != 0).sum().item()
    chunk_ids = inp[:nonpad_len].tolist()
    decoded = tokenizer.DecodeIds(chunk_ids)
    print(f"Chunk {idx}: non-padded length={nonpad_len}")
    print(f"Decoded text: {decoded}")


