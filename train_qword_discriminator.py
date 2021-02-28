import pickle
from collections import defaultdict
from typing import Iterable, Dict, List

import torch
from pytorch_lightning import seed_everything
from sentence_transformers.evaluation import SentenceEvaluator
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn, Tensor

from argparse import ArgumentParser
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.models import Transformer, Pooling
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader


class MySoftmaxLoss(nn.Module):

    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
    ):
        super(MySoftmaxLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels

        self.classifier = nn.Linear(sentence_embedding_dimension, num_labels)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        if type(sentence_features) == list:
            reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features][0]
        else:
            reps = torch.tensor(sentence_features, device=self.model.device)
        vectors_concat = [reps]

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        loss_fct = nn.CrossEntropyLoss()

        if labels is not None:
            loss = loss_fct(output, labels.view(-1))
            return loss
        else:
            return reps, output


class AccuracyEvaluator(SentenceEvaluator):

    def __init__(
            self,
            model: SentenceTransformer,
            softmax_model: nn.Module,
            source_lines: List[str],
            target_lines: List[str],
            label_encoder: LabelEncoder,
            batch_size: int,
            save_best_model: bool,
            monitor_metric: str = "f1_macro"
    ):
        self.model = model
        self.softmax_model = softmax_model
        self.source_lines = source_lines
        self.target_lines = target_lines
        self.label_encoder = label_encoder
        self.batch_size = batch_size

        self.save_best_model = save_best_model
        self.monitor_metric = monitor_metric
        self.best_score = 0

        target_qwords = [line.split()[0].lower() for line in target_lines]
        target_qwords = [qword if not qword.endswith(",") else qword[:-1] for qword in target_qwords]
        self.target_labels = label_encoder.transform(target_qwords)

        self.source_examples = []
        for i, source in enumerate(source_lines):
            self.source_examples += [InputExample(guid=i, texts=[source])]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        print("Start evaluation")

        print("Embedding sources")
        source_embeddings = model.encode(self.source_lines, batch_size=self.batch_size,
                                         show_progress_bar=True, convert_to_numpy=True)

        with torch.no_grad():
            _, prediction = self.softmax_model(source_embeddings, labels=None)

            prediction = torch.argmax(prediction.cpu(), dim=1)

        accuracy = accuracy_score(self.target_labels, prediction)
        f1_macro = f1_score(self.target_labels, prediction, average="macro", zero_division=0)
        f1_micro = f1_score(self.target_labels, prediction, average="micro", zero_division=0)
        print(f"Accuracy = {accuracy} | F1 (micro) = {f1_micro} | F1 (macro) = {f1_macro}\n")

        if self.save_best_model:
            score = 0.0
            if self.monitor_metric == "f1_macro":
                score = f1_macro
            elif self.monitor_metric == "f1_micro":
                score = f1_micro
            elif self.monitor_metric == "acc":
                score = accuracy

            if score > self.best_score:
                print(f"Reaching new best {self.monitor_metric} score: {score}")

                best_model_file = Path(output_path) / "softmax_model.bin"
                print(f"Saving best model to {best_model_file}")
                torch.save(self.softmax_model.state_dict(), best_model_file)

                self.best_score = score

        cl_report_dict = classification_report(self.target_labels, prediction, digits=2, output_dict=True, zero_division=0)
        result_values = [str(epoch), str(round(accuracy, 4)), str(round(f1_micro, 4)), str(round(f1_macro, 4))]
        for i in range(len(self.label_encoder.classes_)):
            label_id = str(i)
            result = "-"
            if label_id in cl_report_dict:
                result = str(round(cl_report_dict[label_id]["f1-score"], 2))

            result_values.append(result)

        result_file = Path(output_path) / "results.tsv"
        if not result_file.exists():
            result_writer = result_file.open("w", encoding="utf8")
            header = ["epoch", "accuracy", "f1_micro", "f1_macro"] + [label for label in self.label_encoder.classes_]
            result_writer.write("\t".join(header) + "\n")
        else:
            result_writer = result_file.open("a", encoding="utf8")

        result_writer.write("\t".join(result_values) + "\n")
        result_writer.close()

        cl_report_str = classification_report(self.target_labels, prediction, zero_division=0)
        cl_report_file = Path(output_path) / "cl_reports.txt"
        cl_report_writer = cl_report_file.open("a", encoding="utf8")
        cl_report_writer.write(f"\n\n{cl_report_str}\n")
        cl_report_writer.close()

        return accuracy


def get_label_encoder():
    qwords = set()
    qwords.update([
        line.strip().split()[0].lower()
        for line in Path("data/task/train.target").open("r").readlines()
    ])

    qwords.update([
        line.strip().split()[0].lower()
        for line in Path("data/task/train.target").open("r").readlines()
    ])

    qwords = sorted([word if not word.endswith(",") else word[:-1] for word in qwords])

    encoder = LabelEncoder()
    encoder.fit(qwords)
    return encoder

def oversample_minor_classes(examples: List[InputExample], threshold: float, rate: int):
    qword_to_examples = defaultdict(list)

    for example in examples:
        qword_to_examples[example.label] += [example]

    all_examples = []
    for id, instances in qword_to_examples.items():
        all_examples += instances

        ratio = len(instances) / len(examples)
        if ratio < threshold:
            all_examples += instances * (rate -1)

    return all_examples


def train(
        model: str,
        source_file: Path,
        target_file: Path,
        output_dir: Path,
        pooling: str,
        epochs: int,
        batch_size: int,
        lower_case: bool,
        os_rate: int,
        os_threshold: float
):
    source_lines = [line.strip() for line in source_file.open("r", encoding="utf8").readlines()]
    if lower_case:
        source_lines = [line.lower() for line in source_lines]

    target_lines = [line.strip() for line in target_file.open("r", encoding="utf8").readlines()]

    encoder = get_label_encoder()

    output_dir.mkdir(parents=True, exist_ok=True)
    label_encoder_file = output_dir / "label_encoder.pkl"
    pickle.dump(encoder, label_encoder_file.open("wb"))

    examples = []
    for i, (source, target) in enumerate(zip(source_lines, target_lines)):
        qword = target.split()[0].lower()
        if qword.endswith(","):
            qword = qword[:-1]

        label_id = encoder.transform([qword])[0]

        example = InputExample(guid=i, texts=[source], label=label_id)
        examples.append(example)

    if os_rate is not None:
        examples = oversample_minor_classes(examples, os_threshold, os_rate)

    if pooling == "mean" or pooling == "cls_st":
        model = SentenceTransformer(model)
    elif pooling == "cls":
        transformer_model = Transformer(model)
        pooling = Pooling(
            word_embedding_dimension=transformer_model.get_word_embedding_dimension(),
            pooling_mode_cls_token=True,
            pooling_mode_mean_tokens=False
        )
        model = SentenceTransformer(modules=[transformer_model, pooling])

    train_loss = MySoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=encoder.classes_.size
    )

    train_data_set = SentenceLabelDataset(examples)
    train_dataloader = DataLoader(train_data_set, batch_size=batch_size)

    evaluator = AccuracyEvaluator(model, train_loss, source_lines, target_lines, encoder, batch_size, True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        # evaluation_steps=1000,
        epochs=epochs,
        warmup_steps=100,
        output_path=str(output_dir),
        save_best_model=True,
    )

    last_model_dir = output_dir / "last_model"
    last_model_dir.mkdir(parents=True, exist_ok=True)

    model.save(last_model_dir)

    softmax_model_file = last_model_dir / "softmax_model.bin"
    torch.save(train_loss, softmax_model_file)

    label_encoder_file = last_model_dir / "label_encoder.pkl"
    pickle.dump(encoder, label_encoder_file.open("wb"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source_file", type=Path, required=True)
    parser.add_argument("--target_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--pooling", type=str, default="cls", required=False)
    parser.add_argument("--epochs", type=int, default=5, required=False)
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--cased", type=bool, default=False, required=False)
    parser.add_argument("--os_rate", type=int, default=None, required=False)
    parser.add_argument("--os_threshold", type=float, default=None, required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    args = parser.parse_args()

    seed_everything(args.seed)

    train(
        model=args.model,
        source_file=args.source_file,
        target_file=args.target_file,
        output_dir=args.output_dir,
        pooling=args.pooling,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lower_case=not args.cased,
        os_rate=args.os_rate,
        os_threshold=args.os_threshold
    )
