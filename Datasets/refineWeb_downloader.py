from datasets import load_dataset, load_from_disk
from WorkStation.StreamingDataset import StreamingDataset
from torch.utils.data import DataLoader

subset = load_dataset(
    "tiiuae/falcon-refinedweb",
    split="train[:0.1%]"
)

splits = subset.train_test_split(test_size=0.1, seed=42)
train_map = splits["train"]
val_map   = splits["test"]

train_map.save_to_disk(r"C:\junha\Datasets\RefinedWebtrain")
val_map.  save_to_disk(r"C:\junha\Datasets\RefinedWeb\val")

print("RefinedWeb 서브셋(≈1B 토큰) train/val split 저장 완료. 이후엔 네트워크 불필요")