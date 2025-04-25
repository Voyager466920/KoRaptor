import random
import numpy as np
import sentencepiece as spm

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(r"C:\junha\Git\BFG_2B\Tokenizer\spm_kowiki.model")

lengths = []
with open(r"C:\junha\Git\BFG_2B\Datasets\train.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
sampled = random.sample(lines, 10000)
for line in sampled:
    ids = tokenizer.EncodeAsIds(line.strip())
    lengths.append(len(ids))

print("50th percentile:", np.percentile(lengths, 50))
print("90th percentile:", np.percentile(lengths, 90))
print("95th percentile:", np.percentile(lengths, 95))
print("99th percentile:", np.percentile(lengths, 99))
