from datasets import load_dataset, disable_caching
import sentencepiece as spm
import itertools, os, random

disable_caching()          # 원하면 캐시 비활성화

stream = load_dataset(
    "openwebtext",
    split="train",
    streaming=True,
    trust_remote_code=True,
)

stream = stream.shuffle(buffer_size=10_000, seed=42)

# 2) 샘플 추출해서 SentencePiece 학습
sample_path = "owt_sample.txt"
with open(sample_path, "w", encoding="utf-8") as f:
    for i, row in zip(range(1_000_000), stream):
        text = row["text"].replace("\n", " ")
        f.write(text + "\n")

spm.SentencePieceTrainer.Train(
    input=sample_path,
    model_prefix="spm_owt",
    vocab_size=32000,
    character_coverage=1.0,
    model_type="unigram",
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
)

os.remove(sample_path)
print("SentencePiece 모델 생성 → spm_owt.model / spm_owt.vocab")
