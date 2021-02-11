import argparse
import random
import shutil

from collections import defaultdict
from pathlib import Path
from sentence_transformers.readers import InputExample
from tqdm import tqdm
from typing import List

from utils import calculate_rouge


def save_examples(examples: List[InputExample], output_file: Path):
    with output_file.open("w") as writer:
        for example in examples:
            writer.write(f"{example.guid}\t{example.texts[0]}\t{example.texts[1]}\t{example.label}\n")


def build_negativ_examples(examples: List[InputExample], num_negative: int, sim_metric: str):
    print(f"Building negative instances for {len(examples)} examples")
    all_targets = [ex.texts[1] for ex in examples]
    neg_examples = []

    for example in tqdm(examples, total=len(examples)):
        target = example.texts[1]
        if sim_metric.startswith("rouge"):
            best_rouge = calculate_rouge([target], [target])[sim_metric]

        for neg_target in random.sample(all_targets, num_negative):
            if neg_target == target:
                continue

            score = 0.0
            if sim_metric.startswith("rouge"):
                score = calculate_rouge([neg_target], [target])[sim_metric] / best_rouge

            neg_examples.append(InputExample(guid=example.guid, texts=[example.texts[0], neg_target], label=score))

    return neg_examples


def generate_data_from_gold_standard(data_dir: Path, num_negative: int, sim_metric: str, output_dir: Path):
    train_examples = []
    test_examples = []
    global_id = 0
    test_file = None

    for i in range(10):
        fold_dir = data_dir / f"fold_{i}"

        source_file = fold_dir / f"test.source"
        sources = [text.strip() for text in source_file.open("r").readlines()]

        target_file = fold_dir / f"test.target"
        targets = [text.strip() for text in target_file.open("r").readlines()]

        examples = []
        for source, target in tqdm(zip(sources, targets), total=len(sources)):
            examples.append(InputExample(guid=global_id, texts=[source, target], label=1.0))
            global_id += 1

        if (i + 1) % 10 == 0:
            test_examples += examples
            test_file = target_file
        else:
            train_examples += examples

    train_examples = train_examples + build_negativ_examples(train_examples, num_negative, sim_metric)
    test_examples = test_examples + build_negativ_examples(test_examples, num_negative, sim_metric)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_examples(train_examples, output_dir / f"train.tsv")
    save_examples(test_examples, output_dir / f"test.tsv")
    shutil.copy(test_file, output_dir / "test.target")


def generate_data_from_predictions(
        data_dir: Path,
        pred_dir: Path,
        output_dir: Path,
        sim_metric: str = "rougeL",
        binary_score: bool = False
):
    train_examples = []
    test_examples = []

    test_file = None
    global_id = 0

    for i in range(10):
        prediction_file = pred_dir / f"fold_{i}.txt.all"
        if not prediction_file.exists():
            print(f"Can't find {prediction_file}")
            continue

        predictions = defaultdict(list)
        for line in prediction_file.open("r"):
            parts = line.split("\t")
            predictions[int(parts[0])] += [parts[1].strip()]

        test_source_file = data_dir / f"fold_{i}" / "test.source"
        test_sources = [line.strip() for line in test_source_file.open("r").readlines()]

        test_target_file = data_dir / f"fold_{i}" / "test.target"
        test_targets = [line.strip() for line in test_target_file.open("r").readlines()]

        print(f"Building examples from {prediction_file}")

        examples = []
        for j, (source, target) in tqdm(enumerate(zip(test_sources, test_targets)), total=len(test_targets)):
            #FIXME: Should we also add the gold standard target to the data set???
            #examples.append(InputExample(guid=str(global_id), texts=[source, target], label=1.0))
            best_rouge = calculate_rouge([target], [target])[sim_metric]

            candidates = predictions[j]
            scores = [calculate_rouge([c], [target])[sim_metric] for c in candidates]
            max_score = max(scores)

            for candidate, score in zip(candidates, scores):
                if binary_score:
                    score = 1.0 if score == max_score else 0.0
                else:
                    score = score / best_rouge

                examples.append(InputExample(guid=str(global_id), texts=[source, candidate], label=score))

            global_id += 1

        if (i+1) % 10 == 0:
            test_examples += examples
            test_file = test_target_file
        else:
            train_examples += examples

    output_dir.mkdir(parents=True, exist_ok=True)
    save_examples(train_examples, output_dir / "train.tsv")
    save_examples(test_examples, output_dir / "test.tsv")
    shutil.copy(test_file, output_dir / "test.target")


def generate_data_from_test_prediction(source_file: Path, prediction_file: Path, output_file: Path):
    source_texts = {i: line.strip() for i, line in enumerate(source_file.open("r").readlines())}

    predictions = defaultdict(list)
    for line in prediction_file.open("r"):
        parts = line.split("\t")
        predictions[int(parts[0])] += [parts[1].strip()]

    writer = output_file.open("w", encoding="utf8")
    for i, source in source_texts.items():
        target_candidates = predictions[i]
        if len(target_candidates) == 0:
            raise AssertionError(f"Can't find predictions for row {i}") # Should never happen :-D

        for target in target_candidates:
            writer.write("\t".join([str(i), source, target, "0.0"]) + "\n")
            writer.flush()

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")

    from_gs_data_parser = subparsers.add_parser("from_gs_data")
    from_gs_data_parser.add_argument("--data_dir", type=Path, required=True)
    from_gs_data_parser.add_argument("--output_dir", type=Path, required=True)
    from_gs_data_parser.add_argument("--sim_metric", type=str, default="", required=False)
    from_gs_data_parser.add_argument("--neg", type=int, default=5, required=False)

    from_pred_data_parser = subparsers.add_parser("from_pred_data")
    from_pred_data_parser.add_argument("--data_dir", type=Path, required=True)
    from_pred_data_parser.add_argument("--prediction_dir", type=Path, required=True)
    from_pred_data_parser.add_argument("--output_dir", type=Path, required=True)
    from_pred_data_parser.add_argument("--sim_metric", type=str, default="rougeL", required=False)
    from_pred_data_parser.add_argument("--binary", type=bool, default=False, required=False)

    from_test_data_parser = subparsers.add_parser("from_test_data")
    from_test_data_parser.add_argument("--source_file", type=Path, required=True)
    from_test_data_parser.add_argument("--prediction_file", type=Path, required=True)
    from_test_data_parser.add_argument("--output_file", type=Path, required=True)

    args = parser.parse_args()

    if args.action == "from_gs_data":
        generate_data_from_gold_standard(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            num_negative=args.neg,
            sim_metric=args.sim_metric
        )

    elif args.action == "from_pred_data":
        generate_data_from_predictions(
            data_dir=args.data_dir,
            pred_dir=args.prediction_dir,
            output_dir=args.output_dir,
            sim_metric=args.sim_metric,
            binary_score=args.binary
        )

    elif args.action == "from_test_data":
        generate_data_from_test_prediction(
            source_file=args.source_file,
            prediction_file=args.prediction_file,
            output_file=args.output_file
        )
