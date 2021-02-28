import numpy as np

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from pytorch_lightning import seed_everything
from sklearn.metrics import accuracy_score, f1_score
from typing import List

from train_qword_discriminator import get_label_encoder
from utils import calculate_rouge


def build_mean_predictions(logit_files: List[Path]):
    id_to_logits = defaultdict(list)

    for logit_file in logit_files:
        reader = logit_file.open("r")
        for i, line in enumerate(reader.readlines()):
            logits = np.array([float(value) for value in line.strip().split()])
            id_to_logits[i] += [logits]
        reader.close()

    mean_predictions = []
    for id in sorted(id_to_logits.keys()):
        logits = np.array(id_to_logits[id])
        logits = np.mean(logits, axis=0)

        mean_predictions += [logits]

    return np.array(mean_predictions)


def eval_discriminator_ensemble(
        logit_files: List[Path],
        target_file: Path,
        candidate_file: Path,
        output_file: Path
):
    label_encoder = get_label_encoder()

    mean_prediction = build_mean_predictions(logit_files)
    prediction = np.argmax(mean_prediction, axis=1)
    prediction_labels = label_encoder.inverse_transform(prediction)

    gold_targets = [line.strip() for line in target_file.open("r", encoding="utf8").readlines()]
    target_labels = [target.split()[0].lower() for target in gold_targets]
    target_labels = [label if not label.endswith(",") else label[:-1] for label in target_labels]
    target_labels = label_encoder.transform(target_labels)

    accuracy = accuracy_score(target_labels, prediction)
    f1_macro = f1_score(target_labels, prediction, average="macro")
    f1_micro = f1_score(target_labels, prediction, average="micro")
    print(f"Accuracy = {accuracy} | F1 (micro) = {f1_micro} | F1 (macro) = {f1_macro}\n")

    id_to_candidates = defaultdict(list)
    for line in candidate_file.open("r", encoding="utf8").readlines():
        id, target = line.strip().split("\t")
        id_to_candidates[int(id)] += [target]

    prediction_targets = []
    prediction_first = []
    for i, pred_qword in enumerate(prediction_labels):
        id_candidates = id_to_candidates[i]
        selected_candidate = id_candidates[0] # Take first candidate as fall back

        for candidate in id_candidates:
            qword = candidate.split()[0].lower()
            if qword.endswith(","):
                qword = qword[:-1]

            if qword == pred_qword:
                selected_candidate = candidate
                break

        prediction_targets.append(selected_candidate)
        prediction_first.append(id_candidates[0])

    qword_file = output_file.parent / (str(output_file.name) + ".qwords")
    qword_writer = qword_file.open("w", encoding="utf8")
    for label in prediction_labels:
        qword_writer.write(label + "\n")
    qword_writer.close()

    prediction_writer = output_file.open("w", encoding="utf8")
    for target in prediction_targets:
        prediction_writer.write(target + "\n")
    prediction_writer.close()


    #seed_everything(42)
    #print("Vanilla")
    #print(calculate_rouge(prediction_first, gold_targets, return_precision_and_recall=True))
    #print(calculate_rouge(prediction_first, gold_targets))
    #print()

    seed_everything(42)
    #print("QWord discriminator")
    #print(calculate_rouge(prediction_targets, gold_targets, return_precision_and_recall=True))
    print(calculate_rouge(prediction_targets, gold_targets))
    print()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logit_files", type=Path, nargs="+", required=True)
    parser.add_argument("--target_file", type=Path, required=True)
    parser.add_argument("--candidate_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)

    args = parser.parse_args()

    eval_discriminator_ensemble(
        logit_files=args.logit_files,
        target_file=args.target_file,
        candidate_file=args.candidate_file,
        output_file=args.output_file,
    )
