from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from run_eval_sent_transformer import read_examples


def check_predictions(
        sim_data_file: Path,
        gold_target_file: Path,
        prediction_file: Path,
        output_file: str

):
    examples = read_examples(sim_data_file)

    id_to_examples = defaultdict(list)
    id_to_first = {}
    id_to_best_score = defaultdict(lambda: -1)
    id_to_best = {}

    for example in examples:
        id_to_examples[example.guid] += [example]

        if example.label > id_to_best_score[example.guid]:
            id_to_best_score[example.guid] = example.label
            id_to_best[example.guid] = example

        if example.guid not in id_to_first:
            id_to_first[example.guid] = example

    predictions = [line.strip() for line in prediction_file.open("r", encoding="utf8").readlines()]
    gold_targets = [line.strip() for line in gold_target_file.open("r", encoding="utf8").readlines()]

    stat_results = []
    for i, (gold_target, prediction) in enumerate(zip(gold_targets, predictions)):
        candidates = id_to_examples[i]
        best_candidate = id_to_best[i]

        first_candidate = id_to_first[i]
        first_score = first_candidate.label

        selected_candidate = [c for c in candidates if c.texts[1] == prediction]
        # if len(selected_candidate) != 1 and selected_candidate[0].label > 0:
        #     #print([f"{c.texts[1]} ({c.label})" for c in  selected_candidate])
        #     #raise AssertionError("Strange prediction!!")

        selected_candidate = selected_candidate[0]
        selected_score = selected_candidate.label

        stat_results.append([
            str(i),
            gold_target,
            best_candidate.texts[1],
            first_candidate.texts[1],
            selected_candidate.texts[1],
            str(first_score),
            str(selected_score),
            str(selected_score-first_score)
        ])

    plain_writer = Path(output_file).open("w", encoding="utf8")
    tsv_writer = Path(output_file + ".tsv").open("w", encoding="utf8")
    tsv_writer.write("\t".join(["ID", "Gold", "Best", "First", "Selected", "Score_First", "Score_Selected", "Score_Diff"]) + "\n")

    for result in stat_results:
        plain_writer.write(f"ID            : {result[0]}\n")
        plain_writer.write(f"Gold          : {result[1]}\n")
        plain_writer.write(f"Best          : {result[2]}\n")
        plain_writer.write(f"First         : {result[3]}\n")
        plain_writer.write(f"Selected      : {result[4]}\n")
        plain_writer.write(f"Score First   : {result[5]}\n")
        plain_writer.write(f"Score Selected: {result[6]}\n")
        plain_writer.write(f"Score Diff    : {result[7]}\n\n\n")

        tsv_writer.write("\t".join(result) + "\n")

    plain_writer.close()
    tsv_writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sim_data_file", type=Path, required=True)
    parser.add_argument("--gold_file", type=Path, required=True)
    parser.add_argument("--prediction_file", type=Path, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    check_predictions(
        sim_data_file=args.sim_data_file,
        gold_target_file=args.gold_file,
        prediction_file=args.prediction_file,
        output_file=args.output_file
    )
