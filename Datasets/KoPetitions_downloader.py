import os
from datasets import load_dataset

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

dataset_name = "heegyu/korean-petitions"
dsdict = load_dataset(dataset_name)

# split 확인
print("SPLITS:", list(dsdict.keys()))

if "train" in dsdict and "validation" in dsdict:
    splits = dsdict
    val_key = "validation"
else:
    base_split = "train" if "train" in dsdict else list(dsdict.keys())[0]
    splits = dsdict[base_split].train_test_split(test_size=0.1, seed=42)
    val_key = "test"

base_dir = r"C:\junha\Datasets\KoreanPetitions"
os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "val"), exist_ok=True)

splits["train"].save_to_disk(os.path.join(base_dir, "train"))
splits[val_key].save_to_disk(os.path.join(base_dir, "val"))

print(f"Saved korean-petitions to {base_dir}")
