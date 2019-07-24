from __future__ import print_function

import pytest
import gluonnlp as nlp
from autogluon.task.named_entity_recognition.dataset import Dataset


@pytest.mark.serial
def test_dataset_class():
    with pytest.raises(NotImplementedError):
        _ = Dataset(name='custom_data')

    with pytest.raises(ValueError):
        _ = Dataset(name='conll2003')

    dset = Dataset(name='conll2003',
                   train_path='/conll2003',
                   val_path='/conll2003')
    assert dset.max_sequence_length == 180
    assert dset.indexes_format is not None
    assert isinstance(dset._vocab, nlp.Vocab)

    dset = Dataset(name='wnut2017')
    assert dset.train_path is not None
    assert dset.val_path is not None
    assert dset.max_sequence_length == 200
    assert dset.indexes_format is not None
    assert isinstance(dset._vocab, nlp.Vocab)


@pytest.mark.serial
def test_wnut2017():
    dset = Dataset(name='wnut2017')
    for file_path in (dset.train_path, dset.val_path):
        with open(file_path) as fp:
            data = fp.readlines()
            if 'train' in file_path:
                train = data
            elif 'dev' in file_path:
                valid = data
            else:
                test = data

    assert len(train) == 66124, len(train)
    assert len(valid) == 16742, len(valid)
