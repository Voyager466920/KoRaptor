import os
import json


def simplify_chat_dataset(input_path: str, output_path: str):
    """
    Simplify the AI Hub emotional dialogue dataset by extracting only the human (HS) and system (SS) dialogues.
    Supports both JSONL (one JSON object per line) and a single JSON array file.
    Writes out a JSONL file with {"prompt": ..., "response": ...} entries.
    """
    def process_record(data, outfile):
        talks = data.get('talk', {}).get('content', {})
        hs_keys = sorted(k for k in talks if k.startswith('HS') and talks[k].strip())
        ss_keys = sorted(k for k in talks if k.startswith('SS') and talks[k].strip())
        for hs_key, ss_key in zip(hs_keys, ss_keys):
            prompt = talks[hs_key].strip()
            response = talks[ss_key].strip()
            json_line = json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False)
            outfile.write(json_line + "\n")

    # Read and process
    with open(input_path, 'r', encoding='utf-8') as infile:
        first_char = infile.read(1)
        infile.seek(0)
        records = []
        if first_char == '[':
            # JSON array
            try:
                records = json.load(infile)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON array: {e}")
                return 0
        else:
            # JSONL
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        print("No records found in input file.")
        return 0

    written = 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for data in records:
            before = written
            process_record(data, outfile)
            # Estimate written by counting lines in outfile? Instead rely on increment in process_record
            # For simplicity, assume each call writes at least one
            written += 1
    return written


def main():
    input_path = r"C:\junha\Datasets\KoRaptor_FineTuning\Train_Conv.json"
    output_path = r"C:\junha\Datasets\KoRaptor_FineTuning\train_simplified.jsonl"

    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return

    count = simplify_chat_dataset(input_path, output_path)
    if count:
        print(f"Preprocessing complete. {count} records processed. Output written to '{output_path}'")
    else:
        print("No valid dialogue records processed. Please check input format.")

if __name__ == '__main__':
    main()
