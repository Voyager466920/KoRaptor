import os, json, hashlib, unicodedata
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets, Dataset

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

dataset_name = "maywell/korean_textbooks"
configs = get_dataset_config_names(dataset_name)

def normalize_record(d):
    drop_keys = {"id","source","url","uuid","__index_level_0__"}
    d2 = {k: d[k] for k in d.keys() if k not in drop_keys}
    s = json.dumps(d2, ensure_ascii=False, sort_keys=True)
    s = unicodedata.normalize("NFKC", s).strip()
    s = " ".join(s.split())
    return s

def make_hash(d):
    s = normalize_record(d)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

partials = []
for cfg in configs:
    try:
        dsdict = load_dataset(dataset_name, cfg)
        keys = list(dsdict.keys())
        base_key = "train" if "train" in keys else keys[0]
        ds = dsdict[base_key]
        ds = ds.map(lambda x: {"__dedupe_key": make_hash(x)}, desc=f"hash:{cfg}")
        partials.append(ds)
        print("OK", cfg, len(ds))
    except Exception as e:
        print("ERR", cfg, e)

merged = concatenate_datasets(partials)

seen = set()
def unique_batch(batch):
    keep = []
    for k in batch["__dedupe_key"]:
        if k in seen:
            keep.append(False)
        else:
            seen.add(k)
            keep.append(True)
    return {"keep": keep}

uniq = merged.map(unique_batch, batched=True, with_indices=False, desc="mark-unique")
uniq = uniq.filter(lambda k: k, input_columns=["keep"])
uniq = uniq.remove_columns([c for c in ["__dedupe_key","keep"] if c in uniq.column_names])

out_dir = r"C:\junha\Datasets\KoreanTextbooks_all_dedup"
os.makedirs(out_dir, exist_ok=True)
uniq_splits = uniq.train_test_split(test_size=0.1, seed=42)
uniq_splits["train"].save_to_disk(os.path.join(out_dir, "train"))
uniq_splits["test"].save_to_disk(os.path.join(out_dir, "val"))
print("TOTAL:", len(merged), "UNIQUE:", len(uniq))
print("Saved to", out_dir)
