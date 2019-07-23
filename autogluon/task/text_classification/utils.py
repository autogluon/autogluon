import json
import os
from typing import AnyStr, Any, Collection

import gluonnlp as nlp
from mxnet import gluon


def read_text_from_file(path: AnyStr) -> AnyStr:
    text = []
    for line in open(path, 'r'):
        line = line.lower().strip()
        text.append(line)
    return ' '.join(line for line in text)


def get_dataset_from_files(path: AnyStr,
                           label_folders: Collection[AnyStr] = None) -> (gluon.data.SimpleDataset, Collection[int]):
    """
    Utility method that reads a dataset from files and returns text, label pairs if labels=True
    :param path:
    :param label_folders: Whether labels are needed or not
    :return:
    """

    include_labels = label_folders is not None
    lines = list()
    label_set = set()
    for folder in os.listdir(os.path.join(path)):
        if include_labels and folder not in label_folders:
            continue
        if folder.startswith('.'):
            # Ignore .DS_Store or something similar
            continue

        for filename in os.listdir(os.path.join(path, folder)):
            if not filename.endswith('.txt'):
                continue
            line = read_text_from_file(os.path.join(path, folder, filename))
            if include_labels:
                # TODO: remove hardcoding!
                lbl = 0 if folder == 'neg' else 1
                lines.append([line, lbl])  # Text , Label pair
                label_set.add(lbl)
            else:
                lines.append(line)

    return gluon.data.SimpleDataset(lines), label_set


def get_dataset_from_json_files(path: AnyStr) -> (gluon.data.SimpleDataset, Collection[int]):
    label_set = set()
    if not path.endswith('.json'):
        path = '.'.join([path, 'json'])
    with open(path) as f:
        content = json.load(f)

    for elem in content:
        label_set.add(elem[1])

    lbl_dict = dict([(y, x) for x, y in enumerate(label_set)])

    for elem in content:
        elem[-1] = lbl_dict[elem[-1]]

    return gluon.data.SimpleDataset(content), label_set


def get_dataset_from_tsv_files(path: AnyStr, field_indices: list = None) -> (gluon.data.SimpleDataset, Collection[int]):
    label_set = set()
    dataset = nlp.data.TSVDataset(filename=path, num_discard_samples=1, field_indices=field_indices)

    for _ in dataset:
        label_set.add(_[-1])

    lbl_dict = dict([(y, x) for x, y in enumerate(label_set)])

    for elem in dataset:
        elem[-1] = lbl_dict[elem[-1]]

    return gluon.data.SimpleDataset(dataset), set(lbl_dict.values())


def get_or_else(x: Any, y: Any) -> Any:
    return y if x is None else x


def flatten_dataset(dataset: gluon.data.SimpleDataset,
                    transform_fn: callable) -> Collection[AnyStr]:
    """
    This flattens and compresses the dataset to a single list of each individual words.
    This will be useful to fine-tune a pre-trained LM model on this dataset.
    The fine-tuning method requires this format for the data.
    :param:dataset
    :param: transform_fn
    :return:
    """

    flattened_data = list()
    print("Original LEN", len(dataset))
    for (sent, label) in dataset:
        sent = transform_fn(sent)
        for word in sent.split():
            flattened_data.append(word)
    print("Flattened LEN", len(flattened_data))
    return flattened_data
