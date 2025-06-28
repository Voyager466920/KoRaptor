from datasets import load_dataset
import os

train_fp = r"C:\junha\Datasets\KoreanText\Training\Train_KoText.txt"
test_fp  = r"C:\junha\Datasets\KoreanText\Testing\Test_KoText.txt"

data_files = {"train": train_fp, "test": test_fp}
ds = load_dataset("text", data_files=data_files)

if not os.path.exists("arrow_train_korean"):
    ds["train"].save_to_disk("arrow_train_korean")
if not os.path.exists("arrow_test_korean"):
    ds["test"].save_to_disk("arrow_test_korean")
