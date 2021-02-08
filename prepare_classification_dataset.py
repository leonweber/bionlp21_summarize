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


def generate_data_from_gold_standard(data_dir: Path, num_negative: int, sim_metric: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "test"]

    for split in splits:
        source_file = data_dir / f"{split}.source"
        sources = [text.strip() for text in source_file.open("r").readlines()]

        target_file = data_dir / f"{split}.target"
        targets = [text.strip() for text in target_file.open("r").readlines()]

        gen_examples = []
        for i, (source, target) in tqdm(enumerate(zip(sources, targets)), total=len(sources)):
            gen_examples.append(InputExample(guid=i, texts=[source, target], label=1.0))

            if sim_metric.startswith("rouge"):
                best_rouge = calculate_rouge([target], [target])[sim_metric]

            for neg_target in random.sample(targets, num_negative):
                if neg_target == target:
                    continue

                score = 0.0
                if sim_metric.startswith("rouge"):
                    score = calculate_rouge([neg_target], [target])[sim_metric] / best_rouge

                gen_examples.append(InputExample(guid=i, texts=[source, neg_target], label=score))

        output_file = output_dir / f"{split}.tsv"
        save_examples(gen_examples, output_file)


def generate_data_from_predictions(data_dir: Path, pred_dir: Path, output_dir: Path, sim_metric: str = "rouge2"):
    train_examples = []
    test_examples = []

    test_file = None
    global_id = 0

    for i in range(10):
        prediction_file = pred_dir / f"predictions_{i}.txt.all"
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

            for candidate, score in zip(candidates, scores):
                examples.append(InputExample(guid=str(global_id), texts=[source, candidate], label=(score/best_rouge)))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")

    from_gs_data_parser = subparsers.add_parser("from_gs_data")
    from_gs_data_parser.add_argument("--data_dir", type=Path, required=True)
    from_gs_data_parser.add_argument("--output_dir", type=Path, required=True)
    from_gs_data_parser.add_argument("--sim_metric", type=str, default="", required=False)
    from_gs_data_parser.add_argument("--neg", type=int, default=5, required=False)

    form_pred_data_parser = subparsers.add_parser("from_pred_data")
    form_pred_data_parser.add_argument("--data_dir", type=Path, required=True)
    form_pred_data_parser.add_argument("--pred_dir", type=Path, required=True)
    form_pred_data_parser.add_argument("--output_dir", type=Path, required=True)
    form_pred_data_parser.add_argument("--sim_metric", type=str, default="rouge1", required=False)

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
            pred_dir=args.pred_dir,
            output_dir=args.output_dir,
            sim_metric=args.sim_metric
        )
