import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from run_eval_sent_transformer import read_examples


def check_predictions(
        sim_data_file: Path,
        gold_target_file: Path,
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

    gold_targets = [line.strip() for line in gold_target_file.open("r", encoding="utf8").readlines()]

    num_correct_word_gen = 0
    num_correct_word_best = 0

    num_found_gold_in_candidates = 0
    num_found_gold_min_better = 0
    num_found_gold_max_better = 0

    num_gold_not_found = 0

    predictions_first = []
    predictions_random = []

    for i, gold_target in enumerate(gold_targets):
        gold_word = gold_target.split()[0].lower()

        candidates = id_to_examples[i]
        qword_to_candidate = defaultdict(list)
        for candidate in candidates:
            qword = candidate.texts[1].split()[0].lower()
            qword_to_candidate[qword] += [candidate]

        first_candidate = id_to_first[i]
        gen_word = first_candidate.texts[1].split()[0].lower()
        if gen_word == gold_word:
            num_correct_word_gen += 1

        best_candidate = id_to_best[i]
        best_word = best_candidate.texts[1].split()[0].lower()
        if best_word == gold_word:
            num_correct_word_best += 1

        if gold_word in qword_to_candidate:
            qword_candidates = qword_to_candidate[gold_word]
            predictions_first.append(qword_candidates[0].texts[1])
            predictions_random.append(random.sample(qword_candidates, 1)[0].texts[1])
        else:
            predictions_first.append(first_candidate.texts[1])
            predictions_random.append(first_candidate.texts[1])
            num_gold_not_found += 1


        if gen_word != gold_word:
            found = False
            max_score = 0.0
            min_score = 1.0

            for c in qword_to_candidate[gold_word]:
                found = True

                if c.label > max_score:
                    max_score = c.label

                if c.label < min_score:
                    min_score = c.label

            if found:
                num_found_gold_in_candidates += 1

                if min_score > first_candidate.label:
                    num_found_gold_min_better += 1

                if max_score > first_candidate.label:
                    num_found_gold_max_better += 1

    print(f"Number correct qword in gen : {num_correct_word_gen} ({num_correct_word_gen/len(gold_targets)})")
    print(f"Number correct qword in best: {num_correct_word_best} ({num_correct_word_best/len(gold_targets)})")
    print(f"Number of gold not found    : {num_gold_not_found} ({num_gold_not_found/len(gold_targets)})")
    print()

    incorrect_qwords = len(gold_targets) - num_correct_word_gen

    print("If generated question word is wrong ....")
    print(f"... found gold qword in candidates: {num_found_gold_in_candidates} ({num_found_gold_in_candidates/incorrect_qwords})")
    print(f"... candidate with qword is better: {num_found_gold_min_better} ({num_found_gold_min_better/incorrect_qwords})")
    print(f"... candidate with qword is better: {num_found_gold_max_better} ({num_found_gold_max_better/incorrect_qwords})")

    output_files = {
        "_first.target": predictions_first,
        "_random.target": predictions_random
    }
    for filename, predictions in output_files.items():
        writer = Path(output_file + filename).open("w", encoding="utf8")
        for prediction in predictions:
            writer.write(prediction + "\n")
        writer.flush()
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sim_data_file", type=Path, required=True)
    parser.add_argument("--gold_file", type=Path, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    check_predictions(
        sim_data_file=args.sim_data_file,
        gold_target_file=args.gold_file,
        output_file=args.output_file
    )
