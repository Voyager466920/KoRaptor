import os
from datasets import load_from_disk

# 경로
textbooks_train = r"C:\junha\Datasets\KoreanTextbooks_all_dedup\val"
petitions_train = r"C:\junha\Datasets\KoreanPetitions\val"

# 불러오기
textbooks_ds = load_from_disk(textbooks_train)
petitions_ds = load_from_disk(petitions_train)

# textbooks: text 컬럼만 남기기 (혹시 None이면 제거)
if "text" in textbooks_ds.column_names:
    textbooks_clean = textbooks_ds.filter(lambda x: x["text"] is not None)
    textbooks_clean = textbooks_clean.remove_columns([c for c in textbooks_clean.column_names if c != "text"])
else:
    raise ValueError("textbooks에 text 컬럼이 없음")

# petitions: content -> text로 바꾸기
if "content" in petitions_ds.column_names:
    petitions_clean = petitions_ds.rename_column("content", "text")
    petitions_clean = petitions_clean.filter(lambda x: x["text"] is not None)
    petitions_clean = petitions_clean.remove_columns([c for c in petitions_clean.column_names if c != "text"])
else:
    raise ValueError("petitions에 content 컬럼이 없음")

print("textbooks 샘플 3개:")
for i in range(3):
    print(textbooks_clean[i])

print("\npetitions 샘플 3개:")
for i in range(3):
    print(petitions_clean[i])

# 저장 (정리된 버전)
out_tb = r"C:\junha\Datasets\KoreanTextbooks_all_dedup_clean\val"
out_pt = r"C:\junha\Datasets\KoreanPetitions_clean\val"
os.makedirs(out_tb, exist_ok=True)
os.makedirs(out_pt, exist_ok=True)

textbooks_clean.save_to_disk(out_tb)
petitions_clean.save_to_disk(out_pt)
