import argparse
import shutil
import pandas as pd
import spacy

from nltk import sent_tokenize
from pandas import DataFrame
from scispacy.linking import EntityLinker
from tqdm import tqdm
from pathlib import Path
from urllib.request import urlretrieve
from random import shuffle

def download_quora_data(output_dir: Path):
    url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
    output_file = output_dir / "quora_duplicate_questions.tsv"

    print(f"Downloading quora file from {url}")
    urlretrieve(url, output_file)


def count_sentences(text):
    return len(sent_tokenize(text))


def calc_sentence_diff(row):
    return abs(row["num_sent_1"] - row["num_sent_2"])


def append_has_hunflair_entities(data: DataFrame) -> DataFrame:
    from flair.models import MultiTagger
    from flair.tokenization import SciSpacySentenceSplitter

    hunflair = MultiTagger.load("hunflair")
    splitter = SciSpacySentenceSplitter()

    has_entity = []
    for id, row in tqdm(data.iterrows(), total=len(data)):
        sentences = splitter.split(row["question1"]) + splitter.split(row["question2"])
        hunflair.predict(sentences)

        num_entities = sum([len(sent.get_spans()) for sent in sentences])
        has_entity.append(num_entities > 0)

    data["has_hunflair_entity"] = has_entity
    return data


def append_has_scispacy_entities(data: DataFrame, linker: str, column: str, threshold: float=0.9) -> DataFrame:
    nlp = spacy.load("en_ner_craft_md")

    linker = EntityLinker(resolve_abbreviations=True, name=linker)
    nlp.add_pipe(linker)

    id_to_has_entity = []
    for id, row in tqdm(data.iterrows(), total=len(data)):
        has_entity = False

        for text in [row["question1"], row["question2"]]:
            doc1 = nlp(text)

            for entity in doc1.ents:
                max_score = max([0] + [umls_ent[1] for umls_ent in entity._.kb_ents])
                if max_score > threshold:
                    has_entity = True
                    break

            if has_entity:
                break

        id_to_has_entity.append(has_entity)

    data[column] = id_to_has_entity
    return data


def read_and_filter_questions(input_file: Path) -> DataFrame:
    print("Preparing quora question pairs")

    qpairs = pd.read_csv(input_file, sep="\t")
    print(f"Found {len(qpairs)} in total")

    qpairs = qpairs[qpairs["is_duplicate"] == 1]
    print(f"Found {len(qpairs)} duplicate question pairs")

    qpairs["num_sent_1"] = qpairs["question1"].apply(count_sentences)
    qpairs["num_sent_2"] = qpairs["question2"].apply(count_sentences)
    qpairs["sent_diff"] = qpairs.apply(calc_sentence_diff, axis=1)

    qpairs = qpairs[qpairs["sent_diff"] >= 1]
    print(f"Found {len(qpairs)} pairs with different amounts of sentences")

    qpairs = qpairs[(qpairs["num_sent_1"] == 1) | (qpairs["num_sent_2"] == 1)]
    print(f"Found {len(qpairs)} pairs with one questions with just one sentence")

    # qpairs = append_has_scispacy_entities(qpairs, "umls", "has_umls_entity")
    # umls = qpairs[qpairs["has_umls_entity"]]
    # print(f"Found {len(umls)} pairs with a umls entity")
    #
    # qpairs = append_has_scispacy_entities(qpairs, "mesh", "has_mesh_entity")
    # mesh = qpairs[qpairs["has_mesh_entity"]]
    # print(f"Found {len(mesh)} pairs with a mesh entity")

    # qpairs = append_has_hunflair_entities(qpairs)
    # hunflair = qpairs[qpairs["has_hunflair_entity"]]
    # print(f"Found {len(hunflair)} pairs with a hunflair entity")

    return qpairs


def save_data(data: DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    source_file = output_dir / f"data.source"
    target_file = output_dir / f"data.target"

    source_writer = source_file.open("w")
    target_writer = target_file.open("w")

    for id, row in data.iterrows():
        source = row["question1"]
        target = row["question2"]

        if row["num_sent_1"] < row["num_sent_2"]:
            tmp = source
            source = target
            target = tmp

        source_writer.write(source + "\n")
        target_writer.write(target + "\n")

    source_writer.close()
    target_writer.close()


def extend_folds(folds_dir: Path, extend_data_dir: Path, output_dir: Path):
    ext_data = list(zip(
        (extend_data_dir / "data.source").open("r").readlines(),
        (extend_data_dir / "data.target").open("r").readlines()
    ))

    for fold_dir in folds_dir.iterdir():
        if not fold_dir.is_dir():
            continue

        fold_data = list(zip(
            (fold_dir / "train.source").open("r").readlines(),
            (fold_dir / "train.target").open("r").readlines()
        ))

        all_data = fold_data + ext_data
        shuffle(all_data)

        fold_out_dir = output_dir / fold_dir.name
        fold_out_dir.mkdir(parents=True, exist_ok=True)

        source_writer = (fold_out_dir / "train.source").open("w")
        target_writer = (fold_out_dir / "train.target").open("w")

        for source, target in all_data:
            source_writer.write(source)
            target_writer.write(target)

        source_writer.close()
        target_writer.close()

        shutil.copy(fold_dir / "val.source", fold_out_dir / "val.source")
        shutil.copy(fold_dir / "val.target", fold_out_dir / "val.target")

        shutil.copy(fold_dir / "test.source", fold_out_dir / "test.source")
        shutil.copy(fold_dir / "test.target", fold_out_dir / "test.target")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")
    prepare_parser = subparsers.add_parser("prepare_quora")

    extend_parser = subparsers.add_parser("extend_folds")
    extend_parser.add_argument("--input_dir", type=Path, required=True)

    args = parser.parse_args()

    aug_data_dir = Path("data/augmention")
    aug_data_dir.mkdir(parents=True, exist_ok=True)

    if args.action == "prepare_quora":
        #download_quora_data(aug_data_dir)
        quora_data = read_and_filter_questions(aug_data_dir / "quora_duplicate_questions.tsv")

        # quora_all_dir = aug_data_dir / "quora_all"
        # save_data(quora_data, quora_all_dir, "data")

        quora_hard = quora_data[quora_data["sent_diff"] >= 3]
        quora_hard_dir = aug_data_dir / "quora_hard3"
        save_data(quora_hard, quora_hard_dir)
        print(f"Found {len(quora_hard)} hard question pairs")

        # quora_umls_dir = aug_data_dir / "quora_umls"
        # umls_data = quora_data[quora_data["has_umls_entity"]]
        # save_data(umls_data, quora_umls_dir)
        #
        # quora_mesh_dir = aug_data_dir / "quora_mesh"
        # mesh_data = quora_data[quora_data["has_mesh_entity"]]
        # save_data(umls_data, quora_mesh_dir)

        # quora_hunflair_dir = aug_data_dir / "quora_hunflair"
        # hunflair_data = quora_data[quora_data["has_hunflair_entity"]]
        # save_data(hunflair_data, quora_hunflair_dir)

    elif args.action == "extend_folds":
        input_name = args.input_dir.name

        extend_folds(
            args.input_dir,
            Path("data/augmention/quora_hunflair"),
            Path(f"data/{input_name}_hunflair")
        )

        extend_folds(
            Path("data/splits_s777"),
            Path("data/augmention/quora_umls"),
            Path(f"data/{input_name}_umls")
        )

        extend_folds(
            Path("data/splits_s777"),
            Path("data/augmention/quora_hard"),
            Path(f"data/{input_name}_hard")
        )

        extend_folds(
            Path("data/splits_s777"),
            Path("data/augmention/quora_all"),
            Path(f"data/{input_name}_all")
        )



