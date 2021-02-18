from typing import Dict, Tuple, List

import pandas as pd
import numpy as np


from numpy import quantile
from numpy.ma import mean
from pandas import DataFrame
from sklearn.model_selection import train_test_split, StratifiedKFold
from transformers import AutoTokenizer
from pathlib import Path


def clean_text(test: str) -> str:
    return test.replace("\n", " ").strip()


def infer_question_type(target: str) -> str:
    return target.split()[0].lower()


def norm_question_type(data: DataFrame, min_threshold: int):
    counts = data["question_word"].value_counts()

    def _perform_norm_question_word(word: str) -> str:
        if counts[word] <= min_threshold:
            return "<rare>"
        return word

    return _perform_norm_question_word


def read_and_clean_data(input_file: Path, column_mapping: Dict[str, str], min_threshold: int) -> DataFrame:
    data = pd.read_csv(input_file, sep="\t")
    data = data.rename(columns=column_mapping)
    data["source"] = data["source"].apply(clean_text)
    data["target"] = data["target"].apply(clean_text)

    data["question_word"] = data["target"].apply(infer_question_type)
    data["question_word_norm"] = data["question_word"].apply(norm_question_type(data, min_threshold))

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

def save_texts(data: DataFrame, output_dir: Path, filename_prefix: str):
    source_file = output_dir / f"{filename_prefix}.source"
    target_file = output_dir / f"{filename_prefix}.target"

    source_writer = source_file.open("w")
    target_writer = target_file.open("w")

    for id, row in data.iterrows():
        source_writer.write(row["source"] + "\n")
        target_writer.write(row["target"] + "\n")

    source_writer.close()
    target_writer.close()


def save_dataset(data: DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    train, val = train_test_split(data, test_size=0.5, stratify=data["question_word_norm"])
    splits = {"train": train, "val": val}

    for split, split_data in splits.items():
        save_texts(split_data, output_dir, split)


def generate_and_save_splits(data: DataFrame, seed: int, output_dir: Path):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    for i, (train_idx, test_idx) in enumerate(kfold.split(data, data["question_word_norm"])):
        train_data = data.loc[train_idx]
        test_data = data.loc[test_idx]

        fold_output_dir = output_dir / f"fold_{i}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        save_texts(train_data, fold_output_dir, "train")
        save_texts(test_data, fold_output_dir, "val")
        save_texts(test_data, fold_output_dir, "test")


def read_task_data(label: str, source_file: Path, target_file: Path):
    source_lines = [line.strip() for line in source_file.open("r", encoding="utf8").readlines()]
    target_lines = [line.strip() for line in target_file.open("r", encoding="utf8").readlines()]

    return [(source, target, label) for source, target, label in zip(source_lines, target_lines, [label] * len(source_lines))]


def save_task_data_split(data: List[Tuple[str, str, str]], output_dir: Path, name: str, target_label: str = None):
    source_writer = (output_dir / f"{name}.source").open("w", encoding="utf8")
    target_writer = (output_dir / f"{name}.target").open("w", encoding="utf8")

    for source, target, label in data:
        if target_label is None or target_label == label:
            source_writer.write(source + "\n")
            target_writer.write(target + "\n")

    source_writer.close()
    target_writer.close()


def build_two_stage_data(gen_train_size: int, disc_train_size: int, output_dir: Path):
    train_data = read_task_data("dev", Path("data/task/train.source"), Path("data/task/train.target"))
    val_data = read_task_data("val", Path("data/task/val.source"), Path("data/task/val.target"))

    all_data = train_data + val_data
    all_data_labels = [label for _, _, label in all_data]

    gen_train, other = train_test_split(all_data, train_size=gen_train_size, stratify=all_data_labels)

    other_labels = [label for _, _, label in other]
    disc_train, test_data = train_test_split(other, train_size=disc_train_size, stratify=other_labels)

    gen_data_dir = output_dir / "gen_data"
    gen_data_dir.mkdir(parents=True, exist_ok=True)
    save_task_data_split(gen_train, gen_data_dir, "train")
    save_task_data_split(gen_train, gen_data_dir, "train_dev", "dev")
    save_task_data_split(gen_train, gen_data_dir, "train_val", "val")

    disc_data_dir = output_dir / "disc_data"
    disc_data_dir.mkdir(parents=True, exist_ok=True)
    save_task_data_split(disc_train, disc_data_dir, "train")
    save_task_data_split(disc_train, disc_data_dir, "train_dev", "dev")
    save_task_data_split(disc_train, disc_data_dir, "train_val", "val")

    test_data_dir = output_dir / "test"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    save_task_data_split(test_data, test_data_dir, "test")
    save_task_data_split(test_data, test_data_dir, "test_dev", "dev")
    save_task_data_split(test_data, test_data_dir, "test_val", "val")


if __name__ == "__main__":
    # Read and convert training data
    build_two_stage_data(600, 250, Path("data/combined1"))

    # data = read_and_clean_data(
    #     input_file=Path("data/MeQSum_ACL2019_BenAbacha_Demner-Fushman.csv"),
    #     column_mapping={"CHQ": "source", "Summary": "target"},
    #     min_threshold=10
    # )
    # print_stats(data)
    # save_texts(data, Path("data/task/"), "train")

    #save_dataset(data, Path("data/50_50"))
    #generate_and_save_splits(data, 17, Path("data/splits_s17"))

    # val_data = read_and_clean_data(
    #     input_file=Path("data/MEDIQA2021-Task1-QuestionSummarization-ValidationSet.csv"),
    #     column_mapping={"NLM Question": "source", "Summary": "target"},
    #     min_threshold=10
    # )
    # save_texts(val_data, Path("data/task_val"), "task_val")


