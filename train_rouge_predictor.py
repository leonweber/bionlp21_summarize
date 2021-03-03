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
from utils import calculate_rouge


class RougeEvaluator(SentenceEvaluator):

    def __init__(
            self,
            source_file: Path,
            candidate_file: Path,
            target_file: Path,
            batch_size: int = 16,
            save_predictions: bool = False
    ):
        self.target_lines = [line.strip() for line in target_file.open("r", encoding="utf8").readlines()]

        self.source_lines = [line.strip() for line in source_file.open("r", encoding="utf8").readlines()]
        self.id_to_source = {i: source for i, source in enumerate(self.source_lines)}

        self.val_examples = []
        for line in candidate_file.open("r", encoding="utf8").readlines():
            id, candidate = line.strip().split("\t")
            id = int(id)
            candidate = candidate.strip()

            source = self.id_to_source[id]
            self.val_examples += [InputExample(guid=id, texts=[source, candidate])]

        self.batch_size = batch_size
        self.save_predictions = save_predictions


    def __call__(self, model: CrossEncoder, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        # Run rouge score prediction
        scores = model.predict([example.texts for example in self.val_examples],
                               batch_size=self.batch_size, show_progress_bar=True)

        # Get example with highest score per id
        best_instances = {}
        best_scores = defaultdict(lambda x: -1)

        for example, score in zip(self.val_examples, scores):
            if score > best_scores[example.guid]:
                best_scores[example.guid] = score
                best_instances[example.guid] = example

        prediction = [
            best_instances[id].texts[1]
            for id in sorted(best_instances.keys())
        ]

        score = calculate_rouge(prediction, self.target_lines)

        if output_path is not None:
            output_dir = Path(output_path)
            tsv_file = output_dir / "results.tsv"

            sorted_keys = sorted(score.keys())

            writer = None
            if not tsv_file.exists():
                writer = tsv_file.open("w")
                writer.write("\t".join(["epoch"] + sorted_keys) + "\n")
            else:
                writer = tsv_file.open("a")

            writer.write("\t".join([str(epoch)] + [str(score[key]) for key in sorted_keys]) + "\n")
            writer.close()

            if self.save_predictions:
                pred_file = output_dir / "prediction.txt"
                with pred_file.open("w") as writer:
                    writer.write("\n".join(prediction))

        return score["rougeL"]


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
