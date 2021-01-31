import argparse
import random
import logging
from collections import defaultdict

from pathlib import Path
from typing import Callable, List

from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, LoggingHandler
from sentence_transformers.evaluation import SentenceEvaluator, SequentialEvaluator, BinaryClassificationEvaluator
from sklearn.metrics.pairwise import paired_cosine_distances, paired_manhattan_distances, paired_euclidean_distances
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import calculate_rouge

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)


class RougeEvaluator(SentenceEvaluator):

    def __init__(self, examples: List[InputExample], batch_size: int = 32, show_progress_bar: bool = False):
        self.examples = examples
        self.batch_size = batch_size

        self.sentences1, self.sentences2 = self.extract_sentences(examples)

        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        logger.info("Running rouge evaluation ...")

        embeddings1 = model.encode(self.sentences1, batch_size=self.batch_size,
                                   show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        embeddings2 = model.encode(self.sentences2, batch_size=self.batch_size,
                                   show_progress_bar=self.show_progress_bar, convert_to_numpy=True)

        cosine_scores = 1-paired_cosine_distances(embeddings1, embeddings2)
        #manhattan_distances = paired_manhattan_distances(embeddings1, embeddings2)
        #euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)

        best_scores = defaultdict(lambda: 0)
        best_instances = {}

        for example, score in zip(self.examples, cosine_scores):
            if score > best_scores[example.guid]:
                best_scores[example.guid] = score
                best_instances[example.guid] = example

        best_sources = []
        best_targets = []
        for id in sorted(best_instances.keys()):
            example = best_instances[id]
            best_sources.append(example.texts[0])
            best_targets.append(example.texts[1])

        logger.info(f"Calculating rouge scores of {len(best_sources)} / {len(best_targets)} examples")
        score = calculate_rouge(best_sources, best_targets)

        if output_path is not None:
            tsv_file = Path(output_path) / "rouge_results.tsv"
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

        return score["rougeLsum"]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)

        return cls(sentences1, sentences2, scores, **kwargs)


    def extract_sentences(self, examples: List[InputExample]):
        sentences1 = []
        sentences2 = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])

        return sentences1, sentences2


def generate_train_data(data_dir: Path, num_negative: int, sim_metrik: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "test"]

    for split in splits:
        source_file = data_dir / f"{split}.source"
        sources = [text.strip() for text in source_file.open("r").readlines()]

        target_file = data_dir / f"{split}.target"
        targets = [text.strip() for text in target_file.open("r").readlines()]

        gen_examples = []
        for i, (source, target) in tqdm(enumerate(zip(sources, targets)), total=len(sources)):
            if sim_metrik.startswith("rouge"):
                rouge_best = calculate_rouge([source], [target])[sim_metrik]
            gen_examples.append(InputExample(guid=i, texts=[source, target], label=1.0))

            for neg_target in random.sample(targets, num_negative):
                if neg_target == target:
                    continue

                score = 0.0
                if sim_metrik.startswith("rouge"):
                    score = calculate_rouge([source], [neg_target])[sim_metrik] / rouge_best

                gen_examples.append(InputExample(guid=i, texts=[source, neg_target], label=score))

        output_file = output_dir / f"{split}.tsv"
        with output_file.open("w") as writer:
            for example in gen_examples:
                writer.write(f"{example.guid}\t{example.texts[0]}\t{example.texts[1]}\t{example.label}\n")


def read_examples(input_file: Path, type_func: Callable):
    examples = []

    with input_file.open("r") as reader:
        for line in reader.readlines():
            id, source, target, label = line.split("\t")
            label = type_func(label)

            examples.append(InputExample(guid=id, texts=[source, target], label=label))

    return examples


def train_sentence_transformer(model_name: str, data_dir: Path):
    # Define the model. Either from scratch of by loading a pre-trained model
    model = SentenceTransformer(model_name)

    train_examples = read_examples(data_dir / "train.tsv", float)[:160]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
    train_loss = losses.CosineSimilarityLoss(model)

    test_examples = read_examples(data_dir / "test.tsv", float)[:200]

    test_sentences1 = [ex.texts[0] for ex in test_examples]
    test_sentences2 = [ex.texts[1] for ex in test_examples]
    test_labels = [int(ex.label) for ex in test_examples]
    cl_evaluator = BinaryClassificationEvaluator(test_sentences1, test_sentences2, test_labels, name="Classification")

    evaluators = SequentialEvaluator(
        evaluators=[
            #cl_evaluator,
            RougeEvaluator(test_examples)
        ]
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluators,
        evaluation_steps=len(train_dataloader),
        epochs=10,
        warmup_steps=100,
        output_path="_test/test-model"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()

    generate_train_data(Path("data/splits_s777/fold_0"), 5, "", Path("_test/data"))
    train_sentence_transformer("distilbert-base-nli-mean-tokens", Path("_test/data"))

