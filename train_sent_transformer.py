import argparse
import logging

from collections import defaultdict
from pathlib import Path
from typing import List

from sentence_transformers import InputExample, LoggingHandler, CrossEncoder
from sentence_transformers.evaluation import SentenceEvaluator, SequentialEvaluator
from torch.utils.data import DataLoader
from run_eval_sent_transformer import evaluate, read_examples

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
            batch_size: int = 12,
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

        score, best_targets = evaluate(
            model=model,
            examples=self.examples,
            gold_targets=self.gold_targets,
            batch_size=self.batch_size
        )

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

        #FIXME: Think about - which metric to perform model selection (rougeL is used as evaluation for the tasks)?
        return score["rougeL"]

def train_sentence_transformer(
        model_name: str,
        data_dir: Path,
        output_dir: Path,
        lower_case: bool,
        epochs: int,
        batch_size: int
):
    model = CrossEncoder(model_name, num_labels=1, max_length=512)

    train_examples = read_examples(data_dir / "train.tsv", lower_case)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    test_examples = read_examples(data_dir / "test.tsv", lower_case)
    gold_targets_file = data_dir / "test.target"

    if lower_case:
        gold_targets = [line.strip().lower() for line in gold_targets_file.open("r").readlines()]
    else:
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
        output_path=str(output_dir),
        save_best_model=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20, required=False)
    parser.add_argument("--bs", type=int, default=8, required=False)
    parser.add_argument("--cased", default=False, required=False, action="store_true")

    args = parser.parse_args()

    train_sentence_transformer(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        lower_case=not args.cased,
        epochs=args.epochs,
        batch_size=args.bs
    )
