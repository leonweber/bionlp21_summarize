from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.evaluation import TripletEvaluator, SentenceEvaluator
from sentence_transformers.losses import BatchAllTripletLoss, TripletLoss, BatchHardTripletLoss, \
    BatchHardSoftMarginTripletLoss, BatchSemiHardTripletLoss
from sklearn.metrics.pairwise import paired_cosine_distances, paired_manhattan_distances, paired_euclidean_distances
from torch.utils.data import DataLoader

from utils import calculate_rouge


class TripletRougeEvaluator(SentenceEvaluator):

    def __init__(
        self,
        candidates: List[InputExample],
        gold_targets: List[str],
        batch_size: int
    ):
        self.candidates = candidates
        self.sources = [c.texts[0] for c in candidates]
        self.targets = [c.texts[1] for c in candidates]

        self.id_to_examples = defaultdict(list)
        self.id_to_indexes = defaultdict(list)
        self.id_to_best_score = defaultdict(lambda: -1)

        for i, candidate in enumerate(candidates):
            self.id_to_examples[candidate.guid] += [candidate]
            self.id_to_indexes[candidate.guid] += [i]

            if candidate.label > self.id_to_best_score[candidate.guid]:
                self.id_to_best_score[candidate.guid] = candidate.label

        self.gold_targets = gold_targets
        self.batch_size = batch_size

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        print("Start rouge evaluation")

        print("Embedding sources")
        embeddings_sources = model.encode(self.sources, batch_size=self.batch_size,
                                          show_progress_bar=True, convert_to_numpy=True)

        print("Embedding targets")
        embeddings_targets = model.encode(self.targets, batch_size=self.batch_size,
                                          show_progress_bar=True, convert_to_numpy=True)

        print("Calculating distances")
        distance_scores = {
            "cos": paired_cosine_distances(embeddings_sources, embeddings_targets),
            "man": paired_manhattan_distances(embeddings_sources, embeddings_targets),
            "euc": paired_euclidean_distances(embeddings_sources, embeddings_targets)
        }

        for distance_metric, distances in distance_scores.items():
            targets = []
            num_best = 0
            for id in sorted(self.id_to_examples.keys()):
                examples = self.id_to_examples[id]

                indexes = self.id_to_indexes[id]
                scores = distances[indexes]

                scored_examples = list(zip(examples, scores))
                scored_examples = sorted(scored_examples, key=lambda pair: pair[1])

                best_candidate = scored_examples[0][0]
                targets.append(best_candidate.texts[1])

                # sorted_scores = [c[1] for c in scored_examples]
                # print(f"Scores: {sorted_scores}")
                # print(f"Best score: {scored_examples[0][1]}")

                best_score = self.id_to_best_score[id]
                if best_score > 0 and best_score == best_candidate.label:
                    num_best += 1

            eval_result = calculate_rouge(targets, self.gold_targets)
            eval_result["num_best"] = num_best

            result_writer = (Path(output_path) / f"result_{distance_metric}.txt").open("w")
            for key in sorted(eval_result.keys()):
                value = eval_result[key]
                result_writer.write(f"{key}: {value}\n")
                print(f"{key}: {value}")
            result_writer.close()

            pred_writer = (Path(output_path) / f"prediction_{distance_metric}.target").open("w")
            for target in targets:
                pred_writer.write(target + "\n")
            pred_writer.close()


def read_triples(input_file: Path, lower_case: bool = True) -> List[InputExample]:
    triples = []
    for line in input_file.open("r", encoding="utf8"):
        id, anchor, pos, neg = line.strip().split("\t")
        id = int(id)
        if lower_case:
            anchor, pos, neg = anchor.lower(), pos.lower(), neg.lower()

        triples.append(InputExample(guid=id, texts=[anchor, pos, neg]))

    return triples


def convert_triple_examples_to_label_examples(triple_examples: List[InputExample]):
    id_to_triples = defaultdict(list)
    for example in triple_examples:
        id_to_triples[example.guid].append(example)

    label_examples = []
    global_label_id = 0
    for id in sorted(id_to_triples.keys()):
        triples = id_to_triples[id]

        source = triples[0].texts[0]
        label_examples.append(InputExample(guid=id, texts=[source], label=global_label_id))

        pos_examples = set([t.texts[1] for t in triples])
        for pos_example in pos_examples:
            label_examples.append(InputExample(guid=id, texts=[pos_example], label=global_label_id))

        # Negative examples should get another id
        global_label_id += 1
        neg_examples = set([t.texts[2] for t in triples])
        for neg_example in neg_examples:
            label_examples.append(InputExample(guid=id, texts=[neg_example], label=global_label_id))

        global_label_id += 1

    return label_examples


def train_discriminator(
        model: str,
        train_file: Path,
        # val_triple_file: Path,
        # val_candidate_file: Path,
        # val_gold_target_file: Path,
        output_dir: Path,
        loss: str,
        margin: float,
        epochs: int,
        batch_size: int,
        lower_case: bool
):
    triples = read_triples(train_file, lower_case)
    model = SentenceTransformer(model)

    convert_examples = True
    if loss == "triplet":
        train_loss = TripletLoss(model=model, triplet_margin=margin)
        convert_examples = False
    elif loss == "all":
        train_loss = BatchAllTripletLoss(model=model, margin=margin)
    elif loss == "hard":
        train_loss = BatchHardTripletLoss(model=model, margin=margin)
    elif loss == "hard_sm":
        train_loss = BatchHardSoftMarginTripletLoss(model=model)
    elif loss == "semi":
        train_loss = BatchSemiHardTripletLoss(model=model, margin=margin)
    else:
        raise AssertionError(f"Unknown loss {loss}")

    print(f"Using {train_loss.__class__.__name__} as loss function")

    if convert_examples:
        train_triples = convert_triple_examples_to_label_examples(triples)
    else:
        train_triples = triples

    train_dataset = SentenceLabelDataset(train_triples)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    evaluator = TripletEvaluator.from_input_examples(
        examples=triples,
        show_progress_bar=True,
        batch_size=batch_size
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        #evaluation_steps=1000,
        epochs=epochs,
        warmup_steps=100,
        output_path=str(output_dir),
        save_best_model=True,
    )

    last_model_dir = output_dir / "last_model"
    last_model_dir.mkdir(parents=True, exist_ok=True)
    model.save(last_model_dir)

    # val_triples = read_triples(val_triple_file)
    # triplet_evaluator = TripletEvaluator.from_input_examples(val_triples)
    # model.evaluate(triplet_evaluator)
    #
    # val_examples = read_examples(val_candidate_file)
    # val_gold_targets = [line.strip() for line in val_gold_target_file.open("r", encoding="utf8").readlines()]
    # rouge_evaluator = TripletRougeEvaluator(
    #     candidates=val_examples,
    #     gold_targets=val_gold_targets,
    #     batch_size=batch_size
    # )
    # model.evaluate(rouge_evaluator)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--loss", type=str, default="all", required=False)
    parser.add_argument("--margin", type=float, default=5.0, required=False)
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--epochs", type=int, default=10, required=False)
    parser.add_argument("--cased", type=bool, default=False, required=False)

    args = parser.parse_args()

    train_discriminator(
        model=args.model,
        train_file=args.train_file,
        output_dir=args.output_dir,
        loss=args.loss,
        margin=args.margin,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lower_case=not args.cased
    )
