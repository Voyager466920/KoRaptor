from datasets import load_from_disk
import pandas as pd

# 1) 로컬에서 KorQuAD 불러오기
train_ds = load_from_disk(r"C:\junha\Datasets\KorQuAD\v1.0\train")
val_ds = load_from_disk(r"C:\junha\Datasets\KorQuAD\v1.0\val")

# 2) 전체 개수 확인
print(f"Train examples: {len(train_ds)}")
print(f"Validation examples: {len(val_ds)}")

# 3) 첫 3개 샘플 간단히 출력
for i in range(3):
    ex = train_ds[i]
    print(f"\n=== Train Sample {i} ===")
    print("Question :", ex["question"])
    print("Context  :", ex["context"][:200].replace("\n", " "), "...")
    print("Answer   :", ex["answers"]["text"])

# 4) (선택) pandas DataFrame 으로 살펴보기
df = pd.DataFrame(train_ds[:10])
print("\nDataFrame preview:")
print(df[["question", "context", "answers"]].head())
