from argparse import ArgumentParser
from collections import Counter
from pathlib import Path


def print_distribution(target_file: Path):
    qwords = [line.strip().split()[0].lower() for line in target_file.open("r").readlines()]
    c = Counter()
    c.update(qwords)

    for key, value in c.most_common(len(c)):
        print(f"{key}: {value} ({value/len(qwords)}%)")
    print("\n\n")


if __name__ == "__main__":
    print("Training:")
    print_distribution(Path("data/task/train.target"))

    print("Validation")
    print_distribution(Path("data/task/val.target"))


