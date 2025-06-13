from datasets import load_dataset, Dataset
from itertools import islice

stream = load_dataset(
    "tiiuae/falcon-refinedweb",
    split="train",
    streaming=True,
    trust_remote_code=True,
)

sampled = list(islice(stream, 484_000))


ds = Dataset.from_list(sampled)

splits = ds.train_test_split(test_size=0.1, seed=42)
splits["train"].save_to_disk(r"C:\junha\Datasets\RefinedWeb\300M\train")
splits["test"].save_to_disk(r"C:\junha\Datasets\RefinedWeb\300M\val")

print("로컬에 ≈300M 토큰 서브셋 저장 완료. 이후 load_from_disk만으로 네트워크 불필요")
