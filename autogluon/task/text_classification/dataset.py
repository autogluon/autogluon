import multiprocessing
from typing import AnyStr

import gluonnlp as nlp
from mxnet import gluon

from .transforms import TextDataTransform, BERTDataTransform, ELMODataTransform

__all__ = ['get_dataset', 'transform', 'get_train_data_lengths', 'get_batchify_fn', 'get_batch_sampler',
           'get_transform_train_fn', 'get_transform_val_fn']

_dataset = {'sst_2': nlp.data.SST_2,
            'glue_sst': nlp.data.GlueSST2,
            'glue_mnli': nlp.data.GlueMNLI,
            'glue_mrpc': nlp.data.GlueMRPC
            }


def get_dataset(name, **kwargs):
    """Returns a dataset by name

    Parameters
    ----------
    name : str
        Name of the dataset.

    Returns
    -------
    Dataset
        The dataset.
    """
    name = name.lower()
    if name not in _dataset:
        err_str = '"%s" is not among the following dataset list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_dataset.keys())))
        raise ValueError(err_str)
    dataset = _dataset[name](**kwargs)
    return dataset


def transform(dataset, transform_fn, num_workers=4):
    # The model type is necessary to pre-process it based on the inputs required to the model.
    with multiprocessing.Pool(num_workers) as pool:
        return gluon.data.SimpleDataset(pool.map(transform_fn, dataset))


def get_train_data_lengths(model_name: AnyStr, dataset, num_workers=4):
    with multiprocessing.Pool(num_workers) as pool:
        if 'bert' in model_name:
            return dataset.transform(lambda token_id, length, segment_id, label_id: length,
                                     lazy=False)
        else:
            return dataset.transform(lambda data, label: int(len(data)), lazy=False)


def get_batchify_fn(model_name: AnyStr):
    if 'bert' in model_name:
        return nlp.data.batchify.Tuple(
            nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
            nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(dtype='int32'))
    else:
        return nlp.data.batchify.Tuple(nlp.data.batchify.Pad(axis=0, ret_length=True),
                                       nlp.data.batchify.Stack(dtype='int32'))


def get_batch_sampler(model_name: AnyStr, train_dataset, batch_size, num_workers=4):
    train_data_lengths = get_train_data_lengths(model_name, train_dataset, num_workers)
    return nlp.data.FixedBucketSampler(train_data_lengths, batch_size=batch_size,
                                       shuffle=True,
                                       num_buckets=10, ratio=0)


def get_transform_train_fn(model_name: AnyStr, vocab: nlp.Vocab, max_sequence_length, is_pair, class_labels=None):
    if 'bert' in model_name:
        dataset_transform = BERTDataTransform(tokenizer=nlp.data.BERTTokenizer(vocab=vocab, lower=True),
                                              max_seq_length=max_sequence_length,
                                              pair=is_pair, class_labels=class_labels)
    elif 'elmo' in model_name:
        dataset_transform = ELMODataTransform(vocab, pair=is_pair, max_sequence_length=max_sequence_length)
    else:
        dataset_transform = TextDataTransform(vocab, transforms=[
            nlp.data.ClipSequence(length=max_sequence_length)],
                                              pair=is_pair, max_sequence_length=max_sequence_length)
    return dataset_transform


def get_transform_val_fn(model_name: AnyStr, vocab: nlp.Vocab, max_sequence_length, is_pair, class_labels=None):
    return get_transform_train_fn(model_name, vocab, max_sequence_length, is_pair, class_labels)
