import argparse

from collections import defaultdict
from typing import List, Callable, Union
from sentence_transformers import CrossEncoder
from sentence_transformers.readers import InputExample
from sklearn.metrics import mean_squared_error
from pathlib import Path
from utils import calculate_rouge
import numpy as np
import torch


def read_examples(source_file: Path, prediction_file: Path):
    examples = []

    predictions_by_guid = defaultdict(set)

    with prediction_file.open("r") as f:
        for line in f:
            guid, pred = line.strip().split("\t")
            guid = int(guid)
            predictions_by_guid[guid].add(pred)
    with source_file.open("r") as f:
        for guid, text in enumerate(f):
            for pred in predictions_by_guid[guid]:
                examples.append(InputExample(guid=guid, texts=[text, pred]))
    return examples


def predict(
        models: List[CrossEncoder],
        examples: List[InputExample],
        batch_size: int
):
    all_scores = []
    for model in models:
        all_scores.append(model.predict([example.texts for example in examples], batch_size, activation_fct=torch.tanh))

    scores = np.mean(all_scores, axis=0)

    # Get example with highest score per id
    best_scores = defaultdict(lambda: -10000)
    best_instances = {}

    for example, score in zip(examples, scores):
        if score > best_scores[example.guid]:
            best_scores[example.guid] = score
            best_instances[example.guid] = example

    best_targets = []
    for id in sorted(best_instances.keys()):
        example = best_instances[id]
        pred_target = example.texts[1]
        best_targets.append(pred_target)

    return best_targets


def predict_rouge_predictor(
        models: str,
        source_file: Path,
        prediction_file: Path,
        batch_size: int,
):

    models = [CrossEncoder(m) for m in models]
    examples = read_examples(source_file, prediction_file)

    predictions = predict(
        models=models,
        examples=examples,
        batch_size=batch_size
    )

    with args.output.open("w") as f:
        f.write("\n".join(predictions))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=True, nargs="+")
    parser.add_argument("--source_file", type=Path, required=True)
    parser.add_argument("--prediction_file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--cased", type=bool, default=False, required=False)

    args = parser.parse_args()

    predict_rouge_predictor(
        models=args.models,
        source_file=args.source_file,
        prediction_file=args.prediction_file,
        batch_size=args.batch_size,
    )
