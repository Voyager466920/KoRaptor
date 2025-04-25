import math
import sentencepiece as spm
import torch
from WorkStation.Datasets import Datasets

# 설정
FILE_PATH   = r"C:\junha\Git\BFG_2B\Datasets\train.txt"
MODEL_PATH  = r"C:\junha\Git\BFG_2B\Tokenizer\spm_kowiki.model"
MAX_SEQ_LEN = 1024
STRIDE      = 512

# 토크나이저 초기화
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(MODEL_PATH)

# Datasets 인스턴스 생성
dataset = Datasets(FILE_PATH, tokenizer, max_seq_len=MAX_SEQ_LEN, stride=STRIDE)
print(f"Total chunks in dataset: {len(dataset)}")

# Inspect first 5 chunks
good = []
print("Inspecting first 5 chunks:")
for idx in range(5):
    inp, lbl = dataset[idx]
    nonpad_len = (inp != 0).sum().item()
    chunk_ids = inp[:nonpad_len].tolist()
    decoded = tokenizer.DecodeIds(chunk_ids)
    print(f"Chunk {idx}: non-padded length={nonpad_len}")
    print(f"Decoded text: {decoded}")

# 첫 번째 라인 토큰화 및 슬라이딩 윈도우 예상 개수 계산
with open(FILE_PATH, "r", encoding="utf-8") as f:
    first_line = f.readline().strip()
ids = tokenizer.EncodeAsIds(first_line)
# 유효 윈도우 계산: 최소 2토큰 이상이 될 때까지
starts = [i for i in range(0, len(ids), STRIDE) if len(ids) - i >= 2]
expected_windows = len(starts)
print(f"First line token length      : {len(ids)}")
print(f"Expected windows for first line: {expected_windows}")

# 첫 3개의 청크 정보 확인
print("\nInspecting first 3 chunks:")
for idx in range(min(3, len(dataset))):
    inp, lbl = dataset[idx]
    start = starts[idx] if idx < expected_windows else None
    actual_len = min(MAX_SEQ_LEN, len(ids) - start) if start is not None else 'N/A'
    # non-pad 토큰 추출
    chunk_ids = inp[:actual_len].tolist() if isinstance(actual_len, int) else []
    decoded = tokenizer.DecodeIds(chunk_ids) if chunk_ids else ''
    print(f"Chunk {idx}: start={start}, length={actual_len}")
    print(f"Decoded text: {decoded}\n")
