import pandas as pd
import numpy as np

from pathlib import Path

from numpy import quantile
from numpy.ma import mean
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


def clean_text(test: str) -> str:
    return test.replace("\n", " ").strip()


def infer_question_type(target: str) -> str:
    return target.split()[0].lower()


def norm_question_type(data: DataFrame):
    counts = data["question_word"].value_counts()

    def _perform_norm_question_word(word: str) -> str:
        if counts[word] < 2:
            return "<rare>"
        return word

    return _perform_norm_question_word


def read_and_clean_data(input_file: Path) -> DataFrame:
    data = pd.read_csv(input_file, sep="\t")
    data = data.rename(columns={"CHQ": "source", "Summary": "target"})
    data["source"] = data["source"].apply(clean_text)
    data["target"] = data["target"].apply(clean_text)

    data["question_word"] = data["target"].apply(infer_question_type)
    data["question_word_norm"] = data["question_word"].apply(norm_question_type(data))

    return data


def print_stats(data: DataFrame):
    #print(f"Type distribution:\n{data['question_word_norm'].value_counts(normalize=True)}")

    source_lengths = np.array([len(v.split()) for v in data["source"].values])
    print(f"Source-Tokens min: {min(source_lengths)}")
    print(f"Source-Tokens max: {max(source_lengths)}")
    print(f"Source-Tokens mean: {mean(source_lengths)}")
    print(f"Source-Tokens q75: {quantile(source_lengths, 0.75)}")
    print(f"Source-Tokens q90: {quantile(source_lengths, 0.90)}")
    print(f"Source-Tokens q95: {quantile(source_lengths, 0.95)}")
    print(f"Source-Tokens q98: {quantile(source_lengths, 0.98)}")
    print("\n\n")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    source_lengths = np.array([len(tokenizer(v).data["input_ids"]) for v in data["source"].values])
    print(f"Source-WP min: {min(source_lengths)}")
    print(f"Source-WP max: {max(source_lengths)}")
    print(f"Source-WP mean: {mean(source_lengths)}")
    print(f"Source-WP q75: {quantile(source_lengths, 0.75)}")
    print(f"Source-WP q90: {quantile(source_lengths, 0.90)}")
    print(f"Source-WP q95: {quantile(source_lengths, 0.95)}")
    print(f"Source-WP q98: {quantile(source_lengths, 0.98)}")



def save_dataset(data: DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    train, val = train_test_split(data, test_size=0.1, stratify=data["question_word_norm"])
    splits = {"train": train, "val": val}

    for split, split_data in splits.items():
        source_file = output_dir / f"{split}.source"
        target_file = output_dir / f"{split}.target"

        source_writer = source_file.open("w")
        target_writer = target_file.open("w")

        for id, row in split_data.iterrows():
            source_writer.write(row["source"] + "\n")
            target_writer.write(row["target"] + "\n")

        source_writer.close()
        target_writer.close()


if __name__ == "__main__":
    data = read_and_clean_data(Path("data/MeQSum_ACL2019_BenAbacha_Demner-Fushman.csv"))
    print_stats(data)

    save_dataset(data, Path("data"))
