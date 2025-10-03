import os
import glob
import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import load_from_disk

base_path = r"C:\junha\Datasets\KoRaptor_Pretrain"

def read_first_arrow(root):
    files = sorted(glob.glob(os.path.join(root, "*.arrow")))
    if not files:
        return None, None
    p = files[0]
    with pa.memory_map(p, "r") as f:
        try:
            r = ipc.open_file(f)
        except pa.lib.ArrowInvalid:
            f.seek(0)
            r = ipc.open_stream(f)
        t = r.read_all()
    return t, p

def show_table(title, table, path):
    print(f"\n=== {title} ===")
    print("열린 파일:", path)
    print("컬럼:", table.schema.names)
    print("총 행 수:", table.num_rows)
    n = min(3, table.num_rows)
    for i in range(n):
        print(f"\n샘플 {i+1}:")
        for name in table.schema.names:
            v = table[name][i].as_py()
            if isinstance(v, str):
                v = v.replace("\n", "\\n")[:200]
            print(f"{name}: {v}")

for folder in sorted(os.listdir(base_path)):
    train_dir = os.path.join(base_path, folder, "train")
    if not os.path.isdir(train_dir):
        continue

    table, used = read_first_arrow(train_dir)
    if table is None:
        print(f"\n{folder}/train: *.arrow 없음")
        continue

    show_table(f"{folder}/train (raw)", table, used)

    names = table.schema.names
    if names == ["indices"] or names == ["__index_level_0__"] or (len(names) == 1 and "indice" in names[0]):
        try:
            ds = load_from_disk(train_dir)
        except Exception as e:
            print(f"\n{folder}/train: load_from_disk 실패: {e}")
            continue
        try:
            ds = ds.flatten_indices()
        except Exception:
            pass
        mat_dir = os.path.join(os.path.dirname(train_dir), "train_materialized")
        if not os.path.exists(mat_dir):
            os.makedirs(mat_dir, exist_ok=True)
        ds.save_to_disk(mat_dir)
        t2, u2 = read_first_arrow(mat_dir)
        if t2 is None:
            print(f"\n{folder}/train_materialized: *.arrow 없음")
        else:
            show_table(f"{folder}/train_materialized", t2, u2)
