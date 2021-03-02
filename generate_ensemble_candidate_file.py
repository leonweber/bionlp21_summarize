from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import List


def combine_candidate_files(candidate_files: List[Path], output_file: Path):
    id_to_qwords = defaultdict(dict)
    id_to_candidates = defaultdict(list)

    for candidate_file in candidate_files:
        num_candidates = 0
        for line in candidate_file.open("r", encoding="utf8"):
            id, candidate = line.strip().split("\t")
            id = int(id)
            qword = candidate.split()[0].lower()

            qwords_dict = id_to_qwords[id]
            if qword in qwords_dict:
                continue

            qwords_dict[qword] = True
            id_to_candidates[id] += [candidate]
            num_candidates += 1

        print(f"Took {num_candidates} from {candidate_file}")

    writer = output_file.open("w", encoding="utf8")
    for id in sorted(id_to_candidates.keys()):
        for candidate in id_to_candidates[id]:
            writer.write(f"{id}\t{candidate}\n")

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--candidate_files", type=Path, nargs="+", required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    args = parser.parse_args()

    combine_candidate_files(args.candidate_files, args.output_file)
