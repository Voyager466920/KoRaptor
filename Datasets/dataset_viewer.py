import os
import glob
import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import load_from_disk

dataset_path = r"C:\junha\Datasets\KoRaptor_Pretrain\KoWiki_TrainVal\train"

def open_arrow_first_shard(root):
    arrow_files = sorted(glob.glob(os.path.join(root, "**", "*.arrow"), recursive=True))
    if not arrow_files:
        raise FileNotFoundError("*.arrow 파일을 찾지 못했습니다.")
    path = arrow_files[0]
    with pa.memory_map(path, "r") as source:
        try:
            reader = ipc.open_file(source)
        except pa.lib.ArrowInvalid:
            source.seek(0)
            reader = ipc.open_stream(source)
        table = reader.read_all()
    return table, path

table, used_path = open_arrow_first_shard(dataset_path)
names = table.schema.names

if names == ["indices"] or names == ["__index_level_0__"] or (len(names) == 1 and "indice" in names[0]):
    ds = load_from_disk(dataset_path)
    try:
        ds = ds.flatten_indices()
    except Exception:
        pass
    mat_path = dataset_path.rstrip("\\/") + "_materialized"
    if not os.path.exists(mat_path):
        os.makedirs(mat_path)
    ds.save_to_disk(mat_path)
    table, used_path = open_arrow_first_shard(mat_path)
    names = table.schema.names

print("열린 파일:", used_path)
print("컬럼:", names)
print("총 행 수:", table.num_rows)

n = min(5, table.num_rows)
for i in range(n):
    print(f"\n샘플 {i+1}:")
    for name in names:
        val = table[name][i].as_py()
        if isinstance(val, str):
            val = val.replace("\n", "\\n")[:1000]
        print(f"{name}: {val}")
