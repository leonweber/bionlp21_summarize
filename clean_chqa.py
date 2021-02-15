import argparse
from pathlib import Path
from flair.datasets.biomedical import InternalBioNerDataset, Entity, CoNLLWriter
from flair.tokenization import SegtokSentenceSplitter
import os



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    documents = {}
    entities_per_document = {}

    for txt_file in args.text.glob("*.txt"):
        name = txt_file.with_suffix("").name
        a1_file = args.ann / txt_file.with_suffix(".ann").name

        with txt_file.open(encoding="utf8") as f:
            text = f.read()
            if "MESSAGE:" in text:
                idx_subject_end = text.index("\n")
                idx_message_start = text.index("MESSAGE:")
                len_removed = idx_message_start - idx_subject_end
                subject = text[:idx_subject_end]
                message = text[idx_message_start:]
                documents[name] = subject + " " + message
            else:
                documents[name] = text
                len_removed = None


        with a1_file.open(encoding="utf8") as ann_reader:
            entities = []

            for line in ann_reader:
                fields = line.strip().split("\t")
                if not fields:
                    continue
                if fields[0].startswith("T"):
                    ann_type = fields[1].split()[0]
                    offsets = fields[1].replace(ann_type + " ", "").split(";")
                    for offset in offsets:
                        start, end = offset.split()
                        start = int(start)
                        end = int(end)

                        if not len_removed or end <= idx_subject_end:
                            pass
                        elif idx_subject_end <= end <= idx_message_start:
                            continue
                        elif end >= idx_message_start:
                            start -= len_removed - 1
                            end -= len_removed - 1
                            pass

                        entities.append(
                            Entity(
                                char_span=(start, end), entity_type=ann_type
                            )
                    )
            entities_per_document[name] = entities
        
    dataset = InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)
    writer = CoNLLWriter(SegtokSentenceSplitter())
    os.makedirs(args.out, exist_ok=True)
    writer.write_to_conll(dataset, args.out/"train.conll")