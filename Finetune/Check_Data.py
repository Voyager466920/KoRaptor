from datasets import load_dataset

# "heegyu/hh-rlhf-ko" 데이터셋을 train split으로 로드
dataset = load_dataset("maywell/koVast", split="train")
print(dataset[0])
