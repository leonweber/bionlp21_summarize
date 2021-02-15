import argparse

from collections import defaultdict
from typing import Union
from sentence_transformers import CrossEncoder
from pathlib import Path

from run_eval_sent_transformer import read_examples


def predict_sent_transformer(
        model: Union[str, CrossEncoder],
        input_file: Path,
        output_file: Path,
        batch_size: int,
        lower_case: bool
):
    if type(model) == str:
        model = CrossEncoder(model)

    examples = read_examples(input_file, lower_case)
    input_texts = [example.texts for example in examples]

    scores = model.predict(input_texts, batch_size)

    # Get example with highest score per id
    best_scores = defaultdict(lambda: -1)
    best_instances = {}

    for example, score in zip(examples, scores):
        if score > best_scores[example.guid]:
            best_scores[example.guid] = score
            best_instances[example.guid] = example

    # Write final prediction
    writer = output_file.open("w", encoding="utf8")
    for id in sorted(best_instances.keys()):
        best_example = best_instances[id]
        writer.write(best_example.texts[1] + "\n")
        writer.flush()

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--bs", type=int, default=8, required=False)
    parser.add_argument("--lower_case", default=False, required=False, action="store_true")

    args = parser.parse_args()

    predict_sent_transformer(
        model=args.model,
        input_file=args.input_file,
        output_file=args.output_file,
        batch_size=args.bs,
        lower_case=args.lower_case
    )
