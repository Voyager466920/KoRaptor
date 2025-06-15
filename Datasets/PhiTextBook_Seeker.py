from datasets import load_dataset
import random

def preview_textbooks(n_samples: int = 5):
    # 1. 데이터셋 로드
    ds = load_dataset("open-phi/textbooks", split="train")  # 총 1,795개 샘플 :contentReference[oaicite:0]{index=0}

    # 2. 전체 개수와 컬럼 확인
    print(f"총 샘플 수: {len(ds)}")
    print(f"컬럼 이름: {ds.column_names}\n")

    # 3. 샘플 인덱스 선택 (랜덤)
    indices = random.sample(range(len(ds)), n_samples)

    # 4. 미리보기 출력
    for idx in indices:
        item = ds[idx]
        print(f"--- 샘플 인덱스: {idx} ---")
        for col in ds.column_names:
            text = item[col]
            length = len(text)
            preview = text.replace("\n", " ")[:200] + ("..." if length > 200 else "")
            print(f"{col} (길이: {length}자): {preview}")
        print()

if __name__ == "__main__":
    preview_textbooks(n_samples=5)
