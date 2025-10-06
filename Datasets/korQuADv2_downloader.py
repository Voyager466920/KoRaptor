import os
from datasets import load_dataset

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

dataset_name = "LGCNS/KorQuAD_2.0"
dsdict = load_dataset(dataset_name)

print("SPLITS:", list(dsdict.keys()))

if "train" in dsdict and "validation" in dsdict:
    splits = dsdict
    val_key = "validation"
else:
    base_split = "train" if "train" in dsdict else list(dsdict.keys())[0]
    splits = dsdict[base_split].train_test_split(test_size=0.1, seed=42)
    val_key = "test"

base_dir = r"C:\junha\Datasets\KorQuAD_2_0"
os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "val"), exist_ok=True)

splits["train"].save_to_disk(os.path.join(base_dir, "train"))
splits[val_key].save_to_disk(os.path.join(base_dir, "val"))

print(f"Saved KorQuAD_2.0 to {base_dir}")
