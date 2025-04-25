#!/usr/bin/env python3

input_file = "kowikitext.train.txt"
train_file = "train.txt"
val_file = "val.txt"
val_ratio = 0.02

from pathlib import Path

lines = Path(input_file).read_text(encoding="utf-8").splitlines()
n_val = int(len(lines) * val_ratio)
val_lines = lines[:n_val]
train_lines = lines[n_val:]

Path(train_file).parent.mkdir(parents=True, exist_ok=True)
Path(val_file).parent.mkdir(parents=True, exist_ok=True)

Path(train_file).write_text("\n".join(train_lines), encoding="utf-8")
Path(val_file).write_text("\n".join(val_lines), encoding="utf-8")

print(f"Wrote {len(train_lines)} lines to {train_file}")
print(f"Wrote {len(val_lines)} lines to {val_file}")
