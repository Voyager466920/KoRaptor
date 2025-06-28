import sentencepiece as spm
import os
from datasets import disable_caching, load_from_disk, load_dataset
from konlpy.tag import Mecab

# 토크나이저는 mecab이슈로 colab에서 사용.
disable_caching()

def create_sample_file(dataset_name, split, output_file, max_samples=1_000_000, cache_dir="./datasets"):
    stream = load_dataset(dataset_name, split=split, streaming=True, cache_dir=cache_dir)
    stream = stream.shuffle(buffer_size=10_000, seed=42)
    with open(output_file, "w", encoding="utf-8") as f:
        for i, row in zip(range(max_samples), stream):
            text = row["sentence"].replace("\n", " ").strip()
            if text:
                f.write(text + "\n")

def create_morph_file(input_file, output_file, analyzer):
    with open(input_file, "r", encoding="utf-8") as fr, open(output_file, "w", encoding="utf-8") as fw:
        for line in fr:
            line = line.strip()
            if line:
                toks = analyzer.morphs(line)
                fw.write(" ".join(toks) + "\n")

create_sample_file(
    "heegyu/kowiki-sentences",
    "train",
    output_file="kowiki_sample.txt",
    max_samples=1_000_000,
)

mecab = Mecab()
create_morph_file("kowiki_sample.txt", "kowiki_morph.txt", mecab)

arrow_train = load_from_disk(r"C:\junha\Datasets\KoreanText\Train").to_iterable_dataset()
with open("train_kotext_morph.txt", "w", encoding="utf-8") as fw:
    for row in arrow_train:
        text = row["text"].replace("\n", " ").strip()
        if text:
            toks = mecab.morphs(text)
            fw.write(" ".join(toks) + "\n")

spm.SentencePieceTrainer.Train(
    input=[
        "kowiki_morph.txt",
        "train_kotext_morph.txt"
    ],
    model_prefix="spm_kowiki",
    vocab_size=30000,
    character_coverage=0.9995,
    model_type="bpe",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    normalization_rule_name="nfkc",
)

for fn in ["kowiki_sample.txt", "kowiki_morph.txt", "train_kotext_morph.txt"]:
    try:
        os.remove(fn)
    except FileNotFoundError:
        pass

print("SentencePiece 모델 생성 완료 → spm_kowiki.model / spm_kowiki.vocab")
