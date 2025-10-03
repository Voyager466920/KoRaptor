from datasets import load_from_disk

# 원본 데이터 경로
dataset_path = r"C:\junha\Datasets\KoRaptor_Pretrain\Interview"
# 저장할 경로
out_path = r"C:\junha\Datasets\KoRaptor_Pretrain\Interview_split"

# 불러오기
ds = load_from_disk(dataset_path)

# train/val 분리 (8:2)
split = ds.train_test_split(test_size=0.2, seed=42)

# 저장
split.save_to_disk(out_path)

print("총 샘플 수:", len(ds))
print("train:", len(split["train"]))
print("validation:", len(split["test"]))
print("저장 경로:", out_path)
