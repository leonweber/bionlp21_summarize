import torch.nn as nn
import argparse
from collections import defaultdict

from pathlib import Path
from typing import List

from pytorch_lightning import seed_everything
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.evaluation import SentenceEvaluator, SequentialEvaluator
from torch.nn import ReLU
from torch.utils.data import DataLoader

from run_eval_sent_transformer import read_examples
from train_sent_transformer import RougeEvaluator


def train_sentence_transformer(
        model_name: str,
        input_file: Path,
        gold_targets_file: Path,
        output_dir: Path,
        lower_case: bool,
        epochs: int,
        batch_size: int
):
    model = CrossEncoder(model_name, num_labels=1, max_length=512)

    train_examples = read_examples(input_file, lower_case)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    if lower_case:
        gold_targets = [line.strip().lower() for line in gold_targets_file.open("r").readlines()]
    else:
        gold_targets = [line.strip() for line in gold_targets_file.open("r").readlines()]

    seq_evaluator = SequentialEvaluator(evaluators=[
        RougeEvaluator(train_examples, gold_targets)
    ])

    optimizer_params = {
        "lr": 2e-2,
        "eps": 1e-6,
        "correct_bias": False
    }

    model.fit(
        train_dataloader,
        evaluator=seq_evaluator,
        acitvation_fct=ReLU(),
        epochs=epochs,
        warmup_steps=100,
        output_path=str(output_dir),
        optimizer_params=optimizer_params,
        save_best_model=True,

    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--target_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--epochs", type=int, default=10, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--cased", type=bool, default=False, required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)

    args = parser.parse_args()

    seed_everything(args.seed)

    train_sentence_transformer(
        model_name=args.model,
        input_file=args.input_file,
        gold_targets_file=args.target_file,
        output_dir=args.output_dir,
        lower_case=not args.cased,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
