from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

REPO = "Voyager466920/KoRaptor"

# 1) 커스텀 Config 먼저 내려받기
config = AutoConfig.from_pretrained(
    REPO,
    trust_remote_code=True
)

# 2) 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    REPO,
    config=config,
    trust_remote_code=True,  # ← 여기에 반드시 True
    use_fast=False           # optional: slow tokenizer 사용
)

# 3) 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    REPO,
    config=config,
    trust_remote_code=True
).eval()

# 4) 인퍼런스 테스트
prompt = "안녕하세요, 오늘 기분이 어때?"
inputs = tokenizer(prompt, return_tensors="pt")
out_ids = model.generate(
    **inputs,
    max_new_tokens=30,
    do_sample=True,
    top_p=0.9,
    temperature=0.8
)
print(tokenizer.decode(out_ids[0], skip_special_tokens=True))
