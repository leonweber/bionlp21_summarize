from collections import Counter

import numpy as np

from pathlib import Path

from numpy import quantile


def calculate_target_source_overlap(source_file: Path, target_file: Path):
    sources = source_file.open("r").readlines()
    targets = target_file.open("r").readlines()

    new_word_counter = Counter()
    overlap_scores = []
    for source, target in zip(sources, targets):
        source_words = set(source.lower().split())
        target_words = set(target.lower().split())

        overlap_scores.append(len(target_words.intersection(source_words)) / len(target_words))

        word_diff = target_words.difference(source_words)
        new_word_counter.update(word_diff)

    return np.array(overlap_scores), new_word_counter


def print_overlap_stats(overlap_scores):
    print(f"Min: {overlap_scores.min()}")
    print(f"Max: {overlap_scores.max()}")
    print(f"Mean: {overlap_scores.mean()}")

    print(f"25q: {quantile(overlap_scores, 0.25)}")
    print(f"50q: {quantile(overlap_scores, 0.50)}")
    print(f"75q: {quantile(overlap_scores, 0.75)}")
    print(f"90q: {quantile(overlap_scores, 0.90)}")
    print(f"95q: {quantile(overlap_scores, 0.95)}")

    print()


if __name__ == "__main__":
    train_overlaps, train_wc = calculate_target_source_overlap(
        Path("data/train.source"),
        Path("data/train.target")
    )

    test_overlaps, test_wc = calculate_target_source_overlap(
        Path("data/val.source"),
        Path("data/val.target")
    )

    all_overlaps = np.concatenate([train_overlaps, test_overlaps], axis=0)
    train_wc.update(test_wc)

    print_overlap_stats(all_overlaps)

    print("Most common missing words:")
    for word, no in train_wc.most_common(20):
        print(f"{no}: {word}")
