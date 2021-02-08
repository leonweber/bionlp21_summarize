from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from utils import calculate_rouge


def select_best_prediction(gold_file: Path, multi_prediction_file: Path, output_file: Path):
    gold_entries = [line.strip() for line in gold_file.open("r").readlines()]
    gold_entries = {i:line for i, line in enumerate(gold_entries)}

    best_scores = defaultdict(lambda: -1)
    best_questions = {}

    pred_lines = multi_prediction_file.open("r").readlines()
    for line in tqdm(pred_lines, total=len(pred_lines)):
        parts = line.strip().split("\t")
        id = int(parts[0])
        question = parts[1].strip()

        score = calculate_rouge([question], [gold_entries[id]])["rouge2"]
        if score > best_scores[id]:
            best_scores[id] = score
            best_questions[id] = question

    fout = output_file.open("w", encoding="utf-8")
    for id in sorted(best_questions.keys()):
        fout.write(best_questions[id] + "\n")

    fout.close()


if __name__ == "__main__":
    select_best_prediction(
        Path("data/splits_s17/fold_0/test.source"),
        Path("_test/prediction.txt.all"),
        Path("_test/prediction_best_source.txt"),
    )

    select_best_prediction(
        Path("data/splits_s17/fold_0/test.target"),
        Path("_test/prediction.txt.all"),
        Path("_test/prediction_best_target.txt"),
    )
