import re
from datasets import load_dataset
from itertools import islice

ds = load_dataset("mc4", "ko", split="train", streaming=True)

count = 0
for ex in ds:
    text = ex["text"]
    if "http" in text: continue
    if len(text) < 50: continue
    if re.search("[A-Za-z]", text): continue
    print(count+1, text.replace("\n", " ")[:200], "…")
    count += 1
    if count >= 10: break
