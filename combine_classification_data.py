from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List

from sentence_transformers import InputExample

from prepare_classification_dataset import save_examples
from prepare_triplet_data import save_triples
from run_eval_sent_transformer import read_examples
from train_discriminator import read_triples


def combine_data_sets(data_sets: List[Tuple[Path, Path]], output_dir: Path):
    global_id = 0

    all_targets = []
    all_examples = []
    for target_file, cl_input_file in data_sets:
        source_lines = [line.strip() for line in target_file.open("r", encoding="utf8").readlines()]
        all_targets += source_lines

        examples = read_examples(cl_input_file)
        id_to_examples = defaultdict(list)
        for ex in examples:
            id_to_examples[ex.guid] += [ex]

        for id in sorted(id_to_examples.keys()):
            for ex in id_to_examples[id]:
                all_examples.append(InputExample(guid=global_id, texts=ex.texts, label=ex.label))
            global_id += 1

    save_examples(all_examples, output_dir / "test.tsv")

    writer = (output_dir / "test.target").open("w", encoding="utf-8")
    for line in all_targets:
        writer.write(line + "\n")

    writer.close()


def combine_example_tsv_files(tsv_files: List[Path], output_dir: Path):
    global_id = 0
    all_examples = []

    for tsv_file in tsv_files:
        examples = read_examples(tsv_file)
        id_to_examples = defaultdict(list)
        for ex in examples:
            id_to_examples[ex.guid] += [ex]

        for id in sorted(id_to_examples.keys()):
            for ex in id_to_examples[id]:
                all_examples.append(InputExample(guid=global_id, texts=ex.texts, label=ex.label))
            global_id += 1

    save_examples(all_examples, output_dir / "train.tsv")


def combine_triplet_tsv_files(tsv_files: List[Path], output_file: Path):
    global_id = 0
    all_examples = []

    for tsv_file in tsv_files:
        examples = read_triples(tsv_file)
        id_to_examples = defaultdict(list)
        for ex in examples:
            id_to_examples[ex.guid] += [ex]

        for id in sorted(id_to_examples.keys()):
            for ex in id_to_examples[id]:
                all_examples.append(InputExample(guid=global_id, texts=ex.texts, label=ex.label))
            global_id += 1

    save_triples(all_examples, output_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file1", type=Path, required=True)
    parser.add_argument("--file2", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    args = parser.parse_args()

    combine_triplet_tsv_files([args.file1, args.file2], args.output_file)

    # combine_data_sets(
    #     [
    #         (Path("data/task/val.target"), Path("output/pred_val/cl_input.txt")),
    #         (Path("data/splits_s777/fold_9/test.target"), Path("output/beam_20/bin_data/test.tsv")),
    #     ],
    #     Path("output/pred_val/")
    # )

    # combine_data_sets(
    #     [
    #         (Path("output/sim_data/test.target"), Path("output/sim_data/test.tsv")),
    #         (Path("data/splits_s777/fold_9/test.target"), Path("output/sim_data/sim_data/test.tsv")),
    #     ],
    #     Path("output/sim_data_train_val/")
    # )
    #
    # # combine_tsv_files(
    # #     tsv_files=[Path("output/beam_20/bin_data/train.tsv"), Path("output/beam_20/bin_data/test.tsv")],
    # #     output_dir=Path("output/pred_val/")
    # # )
    #
    # combine_example_tsv_files(
    #     tsv_files=[Path("output/sim_data/sim_data/train.tsv"), Path("output/sim_data/sim_data/test.tsv")],
    #     output_dir=Path("output/sim_data_val_only//")
    # )

