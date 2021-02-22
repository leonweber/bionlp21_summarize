from argparse import ArgumentParser
from pathlib import Path

from sentence_transformers import SentenceTransformer
from run_eval_sent_transformer import read_examples
from train_discriminator import TripletRougeEvaluator

def eval_discriminator(
        model: str,
        input_file: Path,
        gold_target_file: Path,
        output_dir: Path,
        batch_size: int,
        lower_case: bool
):
    model = SentenceTransformer(model)

    # val_triples = read_triples(val_triple_file)
    # triplet_evaluator = TripletEvaluator.from_input_examples(val_triples)
    # model.evaluate(triplet_evaluator)

    val_examples = read_examples(input_file, lower_case)
    val_gold_targets = [line.strip() for line in gold_target_file.open("r", encoding="utf8").readlines()]

    rouge_evaluator = TripletRougeEvaluator(
        candidates=val_examples,
        gold_targets=val_gold_targets,
        batch_size=batch_size
    )

    model.evaluate(rouge_evaluator, str(output_dir))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--gold_target_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=8, required=False)

    parser.add_argument("--cased", type=bool, default=False, required=False)
    args = parser.parse_args()

    eval_discriminator(
        model=args.model,
        input_file=args.input_file,
        gold_target_file=args.gold_target_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        lower_case=not args.cased
    )
