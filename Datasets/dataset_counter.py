import os
import io
import sys

def count_arrow_file(path, pa):
    try:
        with pa.memory_map(path, 'r') as source:
            try:
                reader = pa.ipc.open_file(source)
                return reader.num_rows
            except pa.lib.ArrowInvalid:
                source.seek(0)
                stream = pa.ipc.open_stream(source)
                total = 0
                for batch in stream:
                    total += batch.num_rows
                return total
    except Exception:
        return 0

def count_arrow_in_dir(path, pa):
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            if fn.lower().endswith(".arrow"):
                total += count_arrow_file(os.path.join(root, fn), pa)
    return total

def count_datasets(base_paths):
    try:
        import pyarrow as pa
    except Exception as e:
        raise RuntimeError(f"pyarrow import error: {e}")
    results = {}
    grand_total = 0
    for base in base_paths:
        train_dir = os.path.join(base, "train")
        if not os.path.exists(train_dir):
            results[base] = "Error: train directory not found"
            continue
        c = count_arrow_in_dir(train_dir, pa)
        results[base] = c
        if isinstance(c, int):
            grand_total += c
    return results, grand_total

if __name__ == "__main__":
    dataset_roots = [
        r"C:\junha\Datasets\KoRaptor_Pretrain\KoRaptor_news",
        r"C:\junha\Datasets\KoRaptor_Pretrain\KoreanPetitions_clean",
        r"C:\junha\Datasets\KoRaptor_Pretrain\KoreanText",
        r"C:\junha\Datasets\KoRaptor_Pretrain\KoWiki_TrainVal",
        r"C:\junha\Datasets\KoRaptor_Pretrain\Interview"
    ]
    counts, total = count_datasets(dataset_roots)
    for k, v in counts.items():
        print(f"{k}\\train: {v}")
    print(f"Total examples: {total}")
