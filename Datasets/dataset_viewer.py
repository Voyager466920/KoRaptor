from datasets import load_from_disk

# 저장한 경로 지정
textbooks_path = r"C:\junha\Datasets\KoreanTextbooks_all_dedup\train"
petite_path = r"C:\junha\Datasets\KoreanPetitions\train"

# 불러오기
textbooks_ds = load_from_disk(textbooks_path)
petite_ds = load_from_disk(petite_path)

print("tiny-textbooks 샘플 3개:")
for i in range(3):
    print(textbooks_ds[i])

print("\npetite 샘플 3개:")
for i in range(3):
    print(petite_ds[i])
