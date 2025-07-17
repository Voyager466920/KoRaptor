from datasets import load_dataset, load_from_disk
import os

# (인터넷 연결 필요) KorQuAD v2.1 전체를 내려받아 로컬에 저장
korquad = load_dataset(
    "KETI-AIR/korquad",
    "v2.1",               # ← 여기서 사용할 config 이름을 지정합니다.
    trust_remote_code=True
)

# train/validation split 이 이미 korquad["train"], korquad["dev"] 로 나뉘어 있습니다.
korquad["train"].save_to_disk(r"C:\junha\Datasets\KorQuAD\v2.1\train")
korquad["dev"].save_to_disk(r"C:\junha\Datasets\KorQuAD\v2.1\val")

print("KorQuAD v2.1 데이터 다운로드 및 로컬 저장 완료.")
