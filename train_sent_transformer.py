import argparse
import logging

from collections import defaultdict

from pathlib import Path
from typing import Callable, List

from sentence_transformers import InputExample, LoggingHandler, CrossEncoder
from sentence_transformers.evaluation import SentenceEvaluator, SequentialEvaluator
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from utils import calculate_rouge

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)


class RougeEvaluator(SentenceEvaluator):

    def __init__(
            self,
            examples: List[InputExample],
            gold_targets: List[str],
            batch_size: int = 32,
            save_predictions: bool = False,
            show_progress_bar: bool = False
    ):
        self.examples = examples
        self.sentences = [example.texts for example in examples]
        self.gold_scores = [example.label for example in examples]

        self.best_gold_scores = defaultdict(lambda: -1)
        for example in examples:
            if example.label > self.best_gold_scores[example.guid]:
                self.best_gold_scores[example.guid] = example.label

        self.gold_targets = gold_targets
        self.batch_size = batch_size
        self.save_predictions = save_predictions

        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

    def __call__(self, model: CrossEncoder, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        logger.info("Running rouge evaluation ...")

        scores = model.predict(self.sentences, self.batch_size, apply_softmax=True)

        best_scores = defaultdict(lambda: -1)
        best_instances = {}

        # Get example with highest score per id
        diff_values = []
        for example, score in zip(self.examples, scores):
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

            score = best_scores[id]
            if score == self.best_gold_scores[id]:
                num_best_target += 1

        logger.info(f"Calculating rouge scores of {len(best_targets)} / {len(self.gold_targets)} examples")
        score = calculate_rouge(best_targets, self.gold_targets)

        score["num_best_target"] = num_best_target
        score["mse"] = mean_squared_error(self.gold_scores, scores)

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
            writer.flush()
            writer.close()

            if self.save_predictions:
                pred_file = output_dir / "predictions.txt"
                with pred_file.open("w") as writer:
                    writer.write("\n".join(best_targets))

        #FIXME: Think about - which metric to perform model selection (rouge1, rouge2, .., mse)?
        return score["rougeLsum"]


def read_examples(input_file: Path, type_func: Callable = float):
    examples = []

    with input_file.open("r") as reader:
        for line in reader.readlines():
            id, source, target, label = line.split("\t")
            label = type_func(label)

            examples.append(InputExample(guid=int(id), texts=[source, target], label=label))

    return examples


def train_sentence_transformer(model_name: str, data_dir: Path, output_dir: Path,
                               epochs: int, batch_size: int):
    model = CrossEncoder(model_name)

    train_examples = read_examples(data_dir / "train.tsv")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    test_examples = read_examples(data_dir / "test.tsv")
    gold_targets_file = data_dir / "test.target"
    gold_targets = [line.strip() for line in gold_targets_file.open("r").readlines()]

    seq_evaluator = SequentialEvaluator(evaluators=[
        RougeEvaluator(test_examples, gold_targets)
    ])

    model.fit(
        train_dataloader,
        evaluator=seq_evaluator,
        evaluation_steps=len(train_dataloader),
        epochs=epochs,
        warmup_steps=100,
        output_path=str(output_dir)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20, required=False)
    parser.add_argument("--bs", type=int, default=8, required=False)

    args = parser.parse_args()

    train_sentence_transformer(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.bs
    )
