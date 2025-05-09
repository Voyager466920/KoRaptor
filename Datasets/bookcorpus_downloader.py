from datasets import load_dataset

# cache_dir 에 bookcorpus 전체를 다운로드
_ = load_dataset(
    "bookcorpus",
    split="train",
    streaming=False,
    cache_dir=r"C:\junha\Datasets"
)
