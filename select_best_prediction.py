from argparse import ArgumentParser
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import variance, mean
from tqdm import tqdm

from rouge_cli import calculate_rouge_path
from utils import calculate_rouge


def select_best_prediction(gold_file: Path, multi_prediction_file: Path, output_file: Path):
    gold_entries = [line.strip() for line in gold_file.open("r").readlines()]
    gold_entries = {i:line for i, line in enumerate(gold_entries)}

    best_scores = defaultdict(lambda: -1)
    best_questions = {}

    scores_per_id = defaultdict(list)

    pred_lines = multi_prediction_file.open("r").readlines()
    for line in tqdm(pred_lines, total=len(pred_lines)):
        parts = line.strip().split("\t")
        id = int(parts[0])
        question = parts[1].strip()

        score = calculate_rouge([question], [gold_entries[id]])["rouge2"]
        scores_per_id[id].append(score)

        if score > best_scores[id]:
            best_scores[id] = score
            best_questions[id] = question

    fout = output_file.open("w", encoding="utf-8")
    for id in sorted(best_questions.keys()):
        fout.write(best_questions[id] + "\n")
    fout.close()

    variances = []
    std_devs = []
    dev_bests = []

    for score_list in scores_per_id.values():
        score_list = sorted(score_list, reverse=True)
        dev_bests.append(score_list[0] - score_list[1])

        var = variance(score_list)
        variances.append(var)
        std_devs.append(sqrt(var))

    stats = {
        "mean_var": round(mean(variances), 6),
        "min_var": round(min(variances), 6),
        "max_var": round(max(variances), 6),

        "mean_std_dev": round(mean(std_devs), 6),
        "min_std_dev": round(min(std_devs), 6),
        "max_std_dev": round(max(std_devs), 6),

        "mean_dist_12": round(mean(dev_bests), 6),
        "min_dist_12": round(min(dev_bests), 6),
        "max_dist_12": round(max(dev_bests), 6)
    }
    print(stats)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--target_file", type=Path, required=True)
    parser.add_argument("--candidate_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)

    args = parser.parse_args()

    select_best_prediction(
        gold_file=args.target_file,
        multi_prediction_file=args.candidate_file,
        output_file=args.output_file
    )

    score = calculate_rouge_path(args.output_file, args.target_file)
    print(score)

    # select_best_prediction(
    #     Path("data/splits_s17/fold_0/test.source"),
    #     Path("_test/prediction.txt.all"),
    #     Path("_test/prediction_best_source.txt"),
    # )

    # select_best_prediction(
    #     Path("data/splits_s777/fold_0/test.target"),
    #     Path("_test/prediction.txt.all"),
    #     Path("_test/prediction_best_target.txt"),
    # )
    #
    # score = calculate_rouge_path(Path("_test/prediction.txt"), Path("data/splits_s777/fold_0/test.target"))
    # print(score)
    #
    # score = calculate_rouge_path(Path("_test/prediction_best_target.txt"), Path("data/splits_s777/fold_0/test.target"))
    # print(score)
