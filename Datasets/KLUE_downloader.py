import os
from datasets import load_dataset

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

klue_tasks = ["ynat", "sts", "nli", "ner", "re", "dp", "mrc", "wos"]
base_dir = r"C:\junha\Datasets\KLUE"
os.makedirs(base_dir, exist_ok=True)

for task in klue_tasks:
    print(f"Downloading KLUE task: {task} ...")
    dsdict = load_dataset("klue", task)

    if "train" in dsdict and "validation" in dsdict:
        splits = dsdict
        val_key = "validation"
    else:
        base_split = "train" if "train" in dsdict else list(dsdict.keys())[0]
        splits = dsdict[base_split].train_test_split(test_size=0.1, seed=42)
        val_key = "test"

    task_dir = os.path.join(base_dir, task)
    os.makedirs(os.path.join(task_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(task_dir, "val"), exist_ok=True)

    splits["train"].save_to_disk(os.path.join(task_dir, "train"))
    splits[val_key].save_to_disk(os.path.join(task_dir, "val"))

    print(f"Saved {task} to {task_dir}")

print("All KLUE datasets downloaded and saved successfully.")
