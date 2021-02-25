import argparse
import math
from collections import Counter, defaultdict
from sklearn.utils import resample as resample


def get_interrogative_dist(lines):
    count = Counter([l.split()[0].lower() for l in lines])
    return {k: v/count.most_common()[0][1] for k, v in count.items()}

def resample_by_interrogative(interrogative_to_index, target_dist):
    resampled_indices = []
    n_majority = max(len(i) for i in interrogative_to_index.values())
    for interrogative in interrogative_to_index:
        if interrogative in target_dist:
            target_frac = target_dist[interrogative]
            n_samples = math.ceil(target_frac * n_majority)
            replace = n_samples > len(interrogative_to_index[interrogative])
            resampled_indices += resample(interrogative_to_index[interrogative], replace=replace, n_samples=n_samples)
        else:
            resampled_indices += interrogative_to_index[interrogative]
    return resampled_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--strategy", default="match", choices=["match", "match_exact", "balance"])
    args = parser.parse_args()

    with open(args.val + ".target") as f:
        val_dist = get_interrogative_dist(f.readlines())
    with open(args.train + ".source") as f:
        train_source = f.readlines()
    with open(args.train + ".target") as f:
        train_target = f.readlines()

    if args.strategy == "match_exact":
        source_new = []
        target_new = []
        for i, line in list(enumerate(train_target)):
            line = line.lower()
            interrogative = line.split()[0].lower()
            if interrogative in val_dist:
                source_new.append(train_source[i])
                target_new.append(train_target[i])
        train_target = target_new
        train_source = source_new

    interrogative_to_index = defaultdict(list)
    for i, line in enumerate(train_target):
        interrogative_to_index[line.split()[0].lower()].append(i)
    interrogative_to_index = dict(interrogative_to_index)

    if args.strategy in ["match", "exact_match"]:
        target_dist = val_dist
    elif args.strategy in ["balance"]:
        target_dist = defaultdict(lambda x: 1)

    train_indices = resample_by_interrogative(target_dist=target_dist,
                                              interrogative_to_index=interrogative_to_index)
    train_source = [train_source[i] for i in train_indices]
    train_target = [train_target[i] for i in train_indices]

    with open(args.out + ".source", "w") as f:
        f.write("n".join(train_source))
    with open(args.out + ".target", "w") as f:
        f.write("n".join(train_target))
