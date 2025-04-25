from Korpora import Korpora

# 1) 한번만 다운로드
Korpora.fetch("kowikitext")

# 2) 로드
corpus = Korpora.load("kowikitext")
train_texts = corpus.train.texts   # list[str] 형태

# 3) SentencePiece 학습
#   (종전 pipeline의 train_spm(raw_txt, …) 대신 아래 사용 가능)
with open("kowikitext.train.txt", "w", encoding="utf-8") as f:
    for line in train_texts:
        # line에는 헤더·본문만 담긴 순수 문장
        f.write(line.replace("\n", " ") + "\n")

# 그 다음부터는 기존과 동일하게 train_spm() → tokenize_chunk()로 진행
