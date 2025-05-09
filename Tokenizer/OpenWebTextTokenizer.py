
import sentencepiece as spm
import itertools, os, random
from datasets import disable_caching, load_dataset

disable_caching()

def create_sample_file(dataset_name, split, config_name=None, output_file=None, max_samples=1000000, cache_dir="./datasets"):
    stream = load_dataset(
        dataset_name,
        config_name,
        split=split,
        streaming=True,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    stream = stream.shuffle(buffer_size=10_000, seed=42)
    with open(output_file, "w", encoding="utf-8") as f:
        for i, row in zip(range(max_samples), stream):
            text = row["text"].replace("\n", " ").strip()
            if text:
                f.write(text + "\n")

# BookCorpus와 Wikipedia 샘플 생성
create_sample_file("bookcorpus", "train", output_file="bc_sample.txt", max_samples=1000000)
#create_sample_file("wikipedia", "train", config_name="20220301.en", output_file="wiki_sample.txt", max_samples=1000000)

# SentencePiece 학습
spm.SentencePieceTrainer.Train(
    input=["bc_sample.txt"], #"wiki_sample.txt"도 추후에 추가
    model_prefix="spm_bc",
    vocab_size=35000,
    character_coverage=0.9995,
    model_type="bpe",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    normalization_rule_name="nfkc",
)

# 임시 파일 삭제
os.remove("bc_sample.txt")
#os.remove("wiki_sample.txt")
print("SentencePiece 모델 생성 → spm_bc.model / spm_bc.vocab")