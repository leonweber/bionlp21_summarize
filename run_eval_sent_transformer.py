import argparse

from collections import defaultdict
from typing import List, Callable, Union
from sentence_transformers import CrossEncoder
from sentence_transformers.readers import InputExample
from sklearn.metrics import mean_squared_error
from pathlib import Path
from utils import calculate_rouge


def read_examples(input_file: Path, type_func: Callable = float):
    examples = []

    with input_file.open("r") as reader:
        for line in reader.readlines():
            id, source, target, label = line.split("\t")
            label = type_func(label)

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
            score = best_scores[id]
            if score == best_gold_scores[id]:
                num_best_target += 1

    score = calculate_rouge(best_targets, gold_targets)
    score["num_best_target"] = num_best_target

    gold_scores = [ex.label for ex in examples]
    score["mse"] = mean_squared_error(gold_scores, scores)

    return score, best_targets


def evaluate_sent_transformer(model: Union[str, CrossEncoder], data_dir: Path, batch_size: int):
    if type(model) == str:
        model = CrossEncoder(model)

    examples = read_examples(data_dir / "test.tsv")
    gold_targets = [line.strip() for line in (data_dir / "test.target").open("r").readlines()]

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
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--bs", type=int, default=8, required=False)

    args = parser.parse_args()

    evaluate_sent_transformer(
        model=args.model,
        data_dir=args.data_dir,
        batch_size=args.bs
    )
