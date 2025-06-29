from datasets import load_from_disk, DatasetDict

src_path = r"C:\junha\Datasets\KoWiki_TrainVal"
ds = DatasetDict({
    "train": load_from_disk(f"{src_path}\\train"),
    "val":   load_from_disk(f"{src_path}\\val"),
})

# ‘sentence’ → ‘text’ 로 컬럼 이름 변경
for split in ds:
    ds[split] = ds[split].rename_column("sentence", "text")

ds.save_to_disk(r"C:\\junha\\Datasets\\KoWiki_TrainVal_text")
