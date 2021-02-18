from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from sentence_transformers import InputExample
from typing import List

from run_eval_sent_transformer import read_examples


def generate_triples(source_file: Path, target_file: Path, sim_file: Path):
    source_lines = [line.strip() for line in source_file.open("r", encoding="utf8").readlines()]
    target_lines = [line.strip() for line in target_file.open("r", encoding="utf8").readlines()]

    examples = read_examples(sim_file)
    id_to_examples = defaultdict(list)
    for example in examples:
        id_to_examples[example.guid].append(example)

    triple_examples = []
    for id in sorted(id_to_examples.keys()):
        source = source_lines[id]
        target = target_lines[id]

        candidates = id_to_examples[id]
        candidates = sorted(candidates, key=lambda ex: ex.label, reverse=True)

        pos_candidates = [c.texts[1] for c in candidates[:3]] + [target]
        neg_candidates = [c.texts[1] for c in candidates[-3:]]

        triple_examples += [
            InputExample(guid=id, texts=[source, pos, neg])
            for pos in pos_candidates
            for neg in neg_candidates
        ]

    return triple_examples


def save_triples(triples: List[InputExample], output_file: Path):
    writer = output_file.open("w", encoding="utf8")

    for example in triples:
        writer.write("\t".join([
            str(example.guid),
            example.texts[0],
            example.texts[1],
            example.texts[2]
        ]) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source_file", type=Path, required=True)
    parser.add_argument("--target_file", type=Path, required=True)
    parser.add_argument("--sim_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    args = parser.parse_args()

    triples = generate_triples(
        source_file=args.source_file,
        target_file=args.target_file,
        sim_file=args.sim_file
    )

    save_triples(triples, args.output_file)


