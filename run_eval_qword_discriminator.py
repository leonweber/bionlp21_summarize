import pickle
import torch

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from pytorch_lightning import seed_everything
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score

from train_qword_discriminator import MySoftmaxLoss
from utils import calculate_rouge


def eval_discriminator(
        model_dir: Path,
        source_file: Path,
        target_file: Path,
        candidate_file: Path,
        output_file: Path,
        batch_size: int,
        lower_case: bool = False
):
    model = SentenceTransformer(str(model_dir))

    encoder_file = model_dir / "label_encoder.pkl"
    label_encoder = pickle.load(encoder_file.open("rb"))

    softmax_model = MySoftmaxLoss(model, model.get_sentence_embedding_dimension(), label_encoder.classes_.size)
    softmax_model.load_state_dict(torch.load(model_dir / "softmax_model.bin"))
    model = softmax_model.model

    model.to("cuda:0")
    softmax_model.to("cuda:0")

    source_lines = [line.strip() for line in source_file.open("r", encoding="utf8").readlines()]
    if lower_case:
        source_lines = [line.lower() for line in source_lines]

    gold_targets = [line.strip() for line in target_file.open("r", encoding="utf8").readlines()]
    target_labels = [target.split()[0].lower() for target in gold_targets]
    target_labels = [label if not label.endswith(",") else label[:-1] for label in target_labels]
    target_labels = label_encoder.transform(target_labels)

    source_embeddings = model.encode(
        sentences=source_lines,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    with torch.no_grad():
        _, prediction = softmax_model(torch.tensor(source_embeddings, device=model.device), labels=None)

        prediction = torch.argmax(prediction.cpu(), dim=1)
        prediction_labels = label_encoder.inverse_transform(prediction)

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

    seed_everything(42)
    print("Vanilla")
    print(calculate_rouge(prediction_first, gold_targets))
    print()

    seed_everything(42)
    print("QWord discriminator")
    print(calculate_rouge(prediction_targets, gold_targets))
    print()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source_file", type=Path, required=True)
    parser.add_argument("--target_file", type=Path, required=True)
    parser.add_argument("--candidate_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)

    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--cased", type=bool, default=False, required=False)
    args = parser.parse_args()

    eval_discriminator(
        model_dir=args.model,
        source_file=args.source_file,
        target_file=args.target_file,
        candidate_file=args.candidate_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        lower_case=not args.cased
    )
