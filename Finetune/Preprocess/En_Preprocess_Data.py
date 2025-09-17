import os
import json

def process_conversations(conversations, outfile):
    written = 0
    pending_prompt = None
    for turn in conversations:
        role = turn.get("from")
        text = turn.get("value")
        if not isinstance(text, str) or not text.strip():
            continue
        if role == "human":
            pending_prompt = text.strip()
        elif role == "gpt" and pending_prompt:
            json_line = json.dumps({"prompt": pending_prompt, "response": text.strip()}, ensure_ascii=False)
            outfile.write(json_line + "\n")
            written += 1
            pending_prompt = None
    return written

def process_record(rec, outfile):
    conversations = rec.get("conversations") or []
    return process_conversations(conversations, outfile)

def simplify_chat_dataset(input_path: str, output_path: str) -> int:
    dirpath = os.path.dirname(output_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    total_written = 0
    with open(input_path, "r", encoding="utf-8") as infile:
        first_char = infile.read(1)
        infile.seek(0)
        with open(output_path, "w", encoding="utf-8") as outfile:
            if first_char == "[":
                try:
                    records = json.load(infile)
                except json.JSONDecodeError:
                    return 0
                if not isinstance(records, list):
                    return 0
                for rec in records:
                    if isinstance(rec, dict):
                        total_written += process_record(rec, outfile)
            else:
                for line in infile:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict):
                        total_written += process_record(rec, outfile)
    return total_written

def main():
    input_path = r"C:\junha\Datasets\chatalpaca-20k.json"
    output_path = r"C:\junha\Datasets\chatalpaca-20k-simplified.jsonl"

    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return

    count = simplify_chat_dataset(input_path, output_path)
    if count:
        print(f"Preprocessing complete. {count} prompt/response pairs written to '{output_path}'.")
    else:
        print("No valid prompt/response pairs found. Please check input format.")

if __name__ == "__main__":
    main()
