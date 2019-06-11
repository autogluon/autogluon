from typing import AnyStr, Any, Collection
from collections import namedtuple
import os
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp


def read_text_from_file(path:AnyStr) -> AnyStr:
    text = []
    for line in open(path, 'r'):
        line = line.lower().strip()
        text.append(line)
    return ' '.join(line for line in text)


def get_dataset_from_files(path:AnyStr,
                           label_folders:Collection[AnyStr] = None) -> (gluon.data.SimpleDataset, Collection[int]):
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
                lbl = 0 if folder=='neg' else 1
                lines.append([line, lbl])  # Text , Label pair
                label_set.add(lbl)
            else:
                lines.append(line)

    return gluon.data.SimpleDataset(lines), label_set


def get_dataset_from_json_files(path: AnyStr) -> (gluon.data.SimpleDataset, Collection[int]):
    import json
    label_set = set()
    if not path.endswith('.json'):
        path = '.'.join([path, 'json'])
    with open(path) as f:
        content = json.load(f)

    for elem in content:
        label_set.add(elem[1])

    return gluon.data.SimpleDataset(content), label_set


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
    print ("Original LEN", len(dataset))
    for (sent, label) in dataset:
        sent = transform_fn(sent)
        for word in sent.split():
            flattened_data.append(word)
    print ("Flattened LEN", len(flattened_data))
    return flattened_data

def convert_arrays_to_text(text_vocab, tag_vocab, np_text_ids, np_true_tags, np_pred_tags, np_valid_length):
    """Convert numpy array data into text
    Parameters
    ----------
    np_text_ids: token text ids (batch_size, seq_len)
    np_true_tags: tag_ids (batch_size, seq_len)
    np_pred_tags: tag_ids (batch_size, seq_len)
    np.array: valid_length (batch_size,) the number of tokens until [SEP] token
    Returns
    -------
    List[List[PredictedToken]]:
    """
    TaggedToken = namedtuple('TaggedToken', ['text', 'tag'])
    PredictedToken = namedtuple('PredictedToken', ['text', 'true_tag', 'pred_tag'])

    NULL_TAG = "X"
    predictions = []
    for sample_index in range(np_valid_length.shape[0]):
        sample_len = np_valid_length[sample_index]
        entries = []
        for i in range(1, sample_len - 1):
            token_text = text_vocab.idx_to_token[np_text_ids[sample_index, i]]
            true_tag = tag_vocab.idx_to_token[int(np_true_tags[sample_index, i])]
            pred_tag = tag_vocab.idx_to_token[int(np_pred_tags[sample_index, i])]
            # we don't need to predict on NULL tags
            if true_tag == NULL_TAG:
                last_entry = entries[-1]
                entries[-1] = PredictedToken(text=last_entry.text + token_text,
                                             true_tag=last_entry.true_tag, pred_tag=last_entry.pred_tag)
            else:
                entries.append(PredictedToken(text=token_text, true_tag=true_tag, pred_tag=pred_tag))

        predictions.append(entries)
    return predictions