import os, re, html, json
from datasets import load_from_disk, DatasetDict, Dataset

def strip_html(s):
    if s is None:
        return ""
    s = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", s)
    s = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_space(s):
    return re.sub(r"\s+", " ", s or "").strip()

def clean_wiki_noise(s):
    s = re.sub(r"\[\s*편집\s*\]", "", s)
    s = re.sub(r"\[\d+\]", "", s)
    cut_keys = ["같이 보기", "각주", "분류", "둘러보기 메뉴", "원본 주소", "개인 도구"]
    idxs = [s.find(k) for k in cut_keys if k in s]
    if idxs:
        s = s[:min([i for i in idxs if i >= 0])]
    s = re.sub(r"\s+", " ", s).strip()
    return s

def remap_answer(ctx_clean, ans_text):
    a = normalize_space(ans_text)
    c = ctx_clean
    i = c.find(a)
    if i >= 0:
        return i
    i = c.lower().find(a.lower())
    if i >= 0:
        return i
    a2 = re.sub(r"\s+", "", a)
    c2 = re.sub(r"\s+", "", c)
    i2 = c2.find(a2)
    if i2 >= 0:
        if a2:
            j = 0
            k = 0
            while j < len(c) and k < i2:
                if not c[j].isspace():
                    k += 1
                j += 1
            return j
    return -1

base_dir = r"C:\junha\Datasets\KoRaptor_FineTuning\KorQuAD_2_0"
train = load_from_disk(os.path.join(base_dir, "train"))
val = load_from_disk(os.path.join(base_dir, "val"))

def build_datasets(ds):
    qa_rows = []
    sft_rows = []
    total = 0
    kept = 0
    for ex in ds:
        total += 1
        ctx_html = ex.get("context") or ex.get("raw_html") or ""
        q = ex.get("question","")
        ans = ex.get("answer",{})
        a_text = ans.get("text", ans.get("html_answer_text",""))
        if isinstance(a_text, list):
            a_text = a_text[0] if a_text else ""
        ctx_clean = strip_html(ctx_html)
        ctx_clean = clean_wiki_noise(ctx_clean)
        q_clean = normalize_space(q)
        a_clean = normalize_space(a_text)
        if not ctx_clean or not q_clean or not a_clean:
            continue
        start = remap_answer(ctx_clean, a_clean)
        if start < 0:
            continue
        kept += 1
        qa_rows.append({
            "id": str(ex.get("id","")),
            "context": ctx_clean,
            "question": q_clean,
            "answers": {"text":[a_clean], "answer_start":[start]}
        })
        sft_rows.append({
            "text": f"문맥: {ctx_clean}\n질문: {q_clean}\n정답: {a_clean}"
        })
    return Dataset.from_list(qa_rows), Dataset.from_list(sft_rows), {"total":total,"kept":kept,"drop":total-kept}

qa_train, sft_train, log_tr = build_datasets(train)
qa_val, sft_val, log_va = build_datasets(val)

qa_dict = DatasetDict({"train": qa_train, "val": qa_val})
sft_dict = DatasetDict({"train": sft_train, "val": sft_val})

out_qa = r"C:\junha\Datasets\KoRaptor_FineTuning\KorQuAD_2_0_QA"
out_sft = r"C:\junha\Datasets\KoRaptor_FineTuning\KorQuAD_2_0_SFT"
os.makedirs(os.path.join(out_qa, "train"), exist_ok=True)
os.makedirs(os.path.join(out_qa, "val"), exist_ok=True)
os.makedirs(os.path.join(out_sft, "train"), exist_ok=True)
os.makedirs(os.path.join(out_sft, "val"), exist_ok=True)

qa_dict["train"].save_to_disk(os.path.join(out_qa, "train"))
qa_dict["val"].save_to_disk(os.path.join(out_qa, "val"))
sft_dict["train"].save_to_disk(os.path.join(out_sft, "train"))
sft_dict["val"].save_to_disk(os.path.join(out_sft, "val"))

print("QA_stats:", json.dumps({"train":log_tr, "val":log_va}, ensure_ascii=False, indent=2))
print("Saved_QA:", out_qa)
print("Saved_SFT:", out_sft)
