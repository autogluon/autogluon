import time
import numpy as np
import mxnet as mx

__all__ = [
    'BaseDataset',
    'SplitSampler',
    'SampledDataset',
    'get_split_samplers']


SPLIT_SEED = int(time.time())


class BaseDataset(mx.gluon.data.Dataset):
    # put any sharable dataset methods here
    pass


def get_split_samplers(train_dataset, split_ratio=0.8):
    num_samples = len(train_dataset)
    split_idx = int(num_samples * split_ratio)
    # numpy seed for consistency
    indices = list(range(num_samples))
    np.random.seed(SPLIT_SEED)
    np.random.shuffle(indices)
    train_sampler = SplitSampler(indices[0: split_idx])
    val_sampler = SplitSampler(indices[split_idx:num_samples])
    return train_sampler, val_sampler


class SplitSampler(object):
    """Samples elements from [start, start+length) randomly without replacement.

    Parameters
    ----------
    length : int
        Length of the sequence.
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        indices = self.indices
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.indices)


class SampledDataset(mx.gluon.data.Dataset):
    """Dataset with elements chosen by a sampler"""
    def __init__(self, dataset, sampler):
        self._dataset = dataset
        self._sampler = sampler
        self._indices = list(iter(sampler))

    def __len__(self):
        return len(self._sampler)

    def __getitem__(self, idx):
        return self._dataset[self._indices[idx]]
