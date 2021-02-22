from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from sentence_transformers import InputExample
from typing import List

from run_eval_sent_transformer import read_examples


def generate_triples(source_file: Path, target_file: Path, sim_file: Path, strategy: str):
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

        #FIXME: This is necessary because of this one example for which the model generates only poor candidates!
        if candidates[0].label == 0 and candidates[-1].label == 0:
            continue

        if strategy == "simple":
            pos_candidates = candidates[:3]
            neg_candidates = candidates[-3:]

        elif strategy == "simple2":
            pos_candidates = []
            last_score = 1.0
            for candidate in candidates:
                # We want to have all 1.0 as pos examples
                if candidate.label == 1.0:
                    pos_candidates.append(candidate)
                    continue

                # If we have at least three "1.0"'s -> stop
                if len(pos_candidates) >= 3:
                    break

                # Take all candidates > 0.9
                if candidate.label > 0.9 or (candidate.label > 0.8 and len(pos_candidates) == 1):
                    pos_candidates.append(candidate)
                    last_score = candidate.label

                break

            neg_candidates = []
            for candidate in candidates[len(pos_candidates):]:
                if abs(candidate.label - last_score) > 0.1:
                    neg_candidates.append(candidate)

            if len(neg_candidates) == 0:
                neg_candidates = candidates[-2:] # Just take the last two

            if len(pos_candidates) == 0 or len(neg_candidates) == 0:
                for c in candidates:
                    print(c)
                print("\n\n")
                raise AssertionError("This should never happen")
        else:
            raise AssertionError(f"Unknown strategy {strategy}")

        pos_candidates = [c.texts[1] for c in pos_candidates] + [target] #FIXME: Should we do this???
        neg_candidates = [c.texts[1] for c in neg_candidates]

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

    parser.add_argument("--strategy", type=str, default="simple2", required=False)

    args = parser.parse_args()

    triples = generate_triples(
        source_file=args.source_file,
        target_file=args.target_file,
        sim_file=args.sim_file,
        strategy=args.strategy
    )

    save_triples(triples, args.output_file)


