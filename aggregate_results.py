import argparse
import numpy as np

from collections import defaultdict
from pathlib import Path

def aggregate_results(input_dir: Path, output_dir: Path):
    results = defaultdict(list)

    for file in input_dir.iterdir():
        if not (file.is_file() and str(file.name).startswith("result_")):
            continue

        file_handle = file.open("r")
        for line in file_handle.readlines():
            key, value = line.split(":")
            results[key.strip()] += [value.strip()]

        file_handle.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.tsv"
    with output_file.open("w") as writer:
        writer.write("metric\tmean\tstd_dev\tmin\tmax\tnum_values\n")
        for metric in sorted(results.keys()):
            values = np.array([float(v) for v in results[metric]])
            writer.write(f"{metric}\t{values.mean()}\t{values.std()}\t{values.min()}\t{values.max()}\t{len(values)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    aggregate_results(args.result_dir, args.result_dir)
