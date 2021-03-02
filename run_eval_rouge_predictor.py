import argparse

from collections import defaultdict
from typing import List, Callable, Union
from sentence_transformers import CrossEncoder
from sentence_transformers.readers import InputExample
from sklearn.metrics import mean_squared_error
from pathlib import Path
from utils import calculate_rouge


def read_examples(input_file: Path, lower_case: bool = False, type_func: Callable = float):
    examples = []

    with input_file.open("r") as reader:
        for line in reader.readlines():
            id, source, target, label = line.split("\t")
            label = type_func(label)

            if lower_case:
                examples.append(InputExample(guid=int(id), texts=[source.lower(), target.lower()], label=label))
            else:
                examples.append(InputExample(guid=int(id), texts=[source, target], label=label))

    return examples


def evaluate(
        model: CrossEncoder,
        examples: List[InputExample],
        gold_targets: List[str],
        batch_size: int
):
    scores = model.predict([example.texts for example in examples], batch_size)

    # get best gold scores from labeled test data
    best_gold_scores = defaultdict(lambda: -1)
    for example in examples:
        if example.label > best_gold_scores[example.guid]:
            best_gold_scores[example.guid] = example.label


    # Get example with highest score per id
    best_scores = defaultdict(lambda: -1)
    best_instances = {}

    diff_values = []
    for example, score in zip(examples, scores):
        if score > best_scores[example.guid]:
            best_scores[example.guid] = score
            best_instances[example.guid] = example

        diff_values.append(abs(score - example.label))

    best_targets = []
    num_best_target = 0
    for id in sorted(best_instances.keys()):
        example = best_instances[id]
        pred_target = example.texts[1]
        best_targets.append(pred_target)

        if best_gold_scores is not None:
            score = example.label
            if score == best_gold_scores[id] and best_gold_scores[id] > 0:
                num_best_target += 1

    score = calculate_rouge(best_targets, gold_targets)
    score["num_best_target"] = num_best_target

    gold_scores = [ex.label for ex in examples]
    score["mse"] = mean_squared_error(gold_scores, scores)

    return score, best_targets


def evaluate_rouge_predictor(
        model: Union[str, CrossEncoder],
        input_file: Path,
        gold_target_file: Path,
        batch_size: int,
        lower_case: bool
):
    if type(model) == str:
        model = CrossEncoder(model)

    examples = read_examples(input_file, lower_case)
    gold_targets = [line.strip() for line in gold_target_file.open("r").readlines()]

    eval_result, _ = evaluate(
        model=model,
        examples=examples,
        gold_targets=gold_targets,
        batch_size=batch_size
    )

    print(str(eval_result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--gold_target_file", type=Path, required=True)

    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--cased", type=bool, default=False, required=False)

    args = parser.parse_args()

    evaluate_rouge_predictor(
        model=args.model,
        input_file=args.input_file,
        gold_target_file=args.gold_target_file,
        batch_size=args.batch_size,
        lower_case=not args.cased
    )
