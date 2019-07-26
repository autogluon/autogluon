import json
from typing import AnyStr, Any, Collection

import gluonnlp as nlp
from mxnet import gluon


def get_dataset_from_txt_files(path: AnyStr) -> (gluon.data.SimpleDataset, Collection[int]):
    """
    This assumes the following format in the txt files: LABEL<delimeter>TEXT
    :param path:
    :return:
    """
    if not path.endswith('.txt'):
        raise ValueError('Passed the dataformat as .txt, but the file extension is not .txt. It is {}'.format(path))
    text = []
    label_set = set()
    for line in open(path, 'r'):
        line = line.lower().strip()
        item = line.split(' ', 1)  # Split it into [LABEL, TEXT]
        label_set.add(item[0])
        text.append([item[1], item[0]])  # Adding ([TEXT, LABEL] tuple to list)

    lbl_dict = dict([(y, x) for x, y in enumerate(label_set)])
    for elem in text:
        elem[-1] = lbl_dict[elem[-1]]

    return gluon.data.SimpleDataset(text), label_set


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
