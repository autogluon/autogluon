import json
from typing import AnyStr, Any, Collection

import gluonnlp as nlp
from mxnet import gluon


def get_dataset_from_txt_files(path: AnyStr) -> gluon.data.SimpleDataset:
    """
    This assumes the following format in the txt files: LABEL<delimeter>TEXT
    :param path:
    :return:
    """
    if not path.endswith('.txt'):
        raise ValueError('Passed the dataformat as .txt, but the file extension is not .txt. It is {}'.format(path))
    text = []

    for line in open(path, 'r'):
        line = line.lower().strip()
        item = line.split(' ', 1)  # Split it into [LABEL, TEXT]
        text.append([item[1], item[0]])  # Adding ([TEXT, LABEL] tuple to list)

    return gluon.data.SimpleDataset(text)


def get_dataset_from_json_files(path: AnyStr) -> gluon.data.SimpleDataset:
    if not path.endswith('.json'):
        path = '.'.join([path, 'json'])
    with open(path) as f:
        content = json.load(f)

    return gluon.data.SimpleDataset(content)


def get_dataset_from_tsv_files(path: AnyStr, field_indices: list = None) -> gluon.data.SimpleDataset:
    dataset = nlp.data.TSVDataset(filename=path, num_discard_samples=1, field_indices=field_indices)
    return dataset


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
