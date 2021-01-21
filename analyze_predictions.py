from collections import defaultdict
from pathlib import Path


def check_questions_words(gold_file: Path, pred_file: Path):
    gold = gold_file.open("r").readlines()
    pred = pred_file.open("r").readlines()

    incorrect_words = defaultdict(lambda: 0)
    correct_words = defaultdict(lambda: 0)

    for i in range(len(gold)):
        gold_question = gold[i].split()[0].lower()
        pred_question = pred[i].split()[0].lower()

        if gold_question == pred_question:
            correct_words[f"{gold_question}"] += 1
        else:
            incorrect_words[f"{gold_question}={pred_question}"] += 1

    num_correct = sum(correct_words.values())
    num_incorrect = sum(incorrect_words.values())

    print(f"Num correct: {num_correct} ({num_correct/len(gold)})")
    print(f"Num incorrect: {num_incorrect} ({num_incorrect/len(gold)})")

    print("\n\n")
    print("correct:")
    for k, v in sorted(correct_words.items(), key=lambda item: item[1]):
        print(f"{k}: {v}")

    print("\n\n")
    print("incorrect:")
    for k, v in sorted(incorrect_words.items(), key=lambda item: item[1]):
        print(f"{k}: {v}")


if __name__ == "__main__":
    check_questions_words(
        Path("data/50_50/val.target"),
        Path("predictions_50_50.txt")
    )
