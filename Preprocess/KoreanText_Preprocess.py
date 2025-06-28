import json

with open(r"news_val.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("news_val.txt", "w", encoding="utf-8") as out:
    for doc in data["documents"]:
        for block in doc["text"]:
            for item in block:
                out.write(item["sentence"].strip() + "\n")
