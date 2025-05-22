from datasets import load_dataset

CACHE_DIR = r"C:\junha\Datasets\HF"

# -- WikiText-2 --
ds_wiki2_train = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1",
    split="train",
    cache_dir=CACHE_DIR
)
ds_wiki2_val = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1",
    split="validation",
    cache_dir=CACHE_DIR
)

# 로컬 디스크에 저장할 경로
ds_wiki2_train.save_to_disk(r"C:\junha\Datasets\WikiText2\train")
ds_wiki2_val.save_to_disk(r"C:\junha\Datasets\WikiText2\val")

# -- WikiText-103 --
ds_wiki103_train = load_dataset(
    "wikitext",
    "wikitext-103-raw-v1",
    split="train",
    cache_dir=CACHE_DIR
)
ds_wiki103_val = load_dataset(
    "wikitext",
    "wikitext-103-raw-v1",
    split="validation",
    cache_dir=CACHE_DIR
)

ds_wiki103_train.save_to_disk(r"C:\junha\Datasets\WikiText103\train")
ds_wiki103_val.save_to_disk(r"C:\junha\Datasets\WikiText103\val")

print("✅ WikiText-2 / WikiText-103 다운로드 및 로컬 저장 완료. 이후 streaming=True, local_files_only=True 설정하시면 네트워크 없이 바로 사용 가능합니다.")
