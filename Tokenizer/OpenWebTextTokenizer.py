import sentencepiece as spm
import os
from datasets import disable_caching, load_dataset

disable_caching()

def create_sample_file(dataset_name, split, config_name=None, output_file=None, max_samples=1_000_000, cache_dir="./datasets"):
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

create_sample_file(
    "bookcorpus",
    "train",
    output_file="bc_sample.txt",
    max_samples=1_000_000,
)

create_sample_file(
    "wikitext",
    "train",
    config_name="wikitext-2-raw-v1",
    output_file="wiki2_sample.txt",
    max_samples=200_000,
)

create_sample_file(
    "wikitext",
    "train",
    config_name="wikitext-103-raw-v1",
    output_file="wiki103_sample.txt",
    max_samples=1_000_000,
)

spm.SentencePieceTrainer.Train(
    input=["bc_sample.txt", "wiki2_sample.txt", "wiki103_sample.txt"],
    model_prefix="spm_bc",
    vocab_size=50_000,
    character_coverage=0.9995,
    model_type="bpe",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    normalization_rule_name="nfkc",
)
for tmp in ["bc_sample.txt", "wiki2_sample.txt", "wiki103_sample.txt"]:
    try:
        os.remove(tmp)
    except FileNotFoundError:
        pass

print("SentencePiece 모델 생성 완료 → spm_bc.model / spm_bc.vocab")
