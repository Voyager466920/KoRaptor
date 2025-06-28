import os
from datasets import load_dataset, Dataset

# Stream full KoWiki sentences dataset
stream = load_dataset(
    "heegyu/kowiki-sentences",
    split="train",
    streaming=True,
)

# Collect all samples
subset = []
for sample in stream:
    text = sample["sentence"].replace("\n", " ").strip()
    if text:
        subset.append({"sentence": text})

# Convert to a Dataset and split into train/validation (90/10)
new_ds = Dataset.from_list(subset)
splits = new_ds.train_test_split(test_size=0.1, seed=42)

# Save splits to disk
os.makedirs(r"C:\junha\Datasets\KoWiki_TrainVal\train", exist_ok=True)
os.makedirs(r"C:\junha\Datasets\KoWiki_TrainVal\val", exist_ok=True)
splits["train"].save_to_disk(r"C:\junha\Datasets\KoWiki_TrainVal\train")
splits["test"].save_to_disk(r"C:\junha\Datasets\KoWiki_TrainVal\val")

print("KoWiki sentences dataset (no token limit) split and saved.")
