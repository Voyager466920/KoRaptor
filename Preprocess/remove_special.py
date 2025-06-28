import re

def clean_file(input_path, output_path):
    # 패턴에 걸리는 라인은 쓰지 않고 건너뜁니다.
    pattern = re.compile(r'[@&]')
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            if pattern.search(line):
                continue  # 특수문자 포함된 라인 스킵
            fout.write(line)

if __name__ == '__main__':
    input_path  = r"news_val.txt"
    output_path = r"news_val_cleaned.txt"

    clean_file(input_path, output_path)
    print(f"전처리 완료: {output_path}")
