from __future__ import print_function

import pytest
import os
from mxnet.gluon.utils import download
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@pytest.mark.serial
def test_wnut2017():
    # TODO: This method changes when WNUT2017 is part of GluonNLP dataset
    url_format = 'https://noisy-text.github.io/2017/files/{}'
    train_filename = 'wnut17train.conll'
    val_filename = 'emerging.dev.conll'
    test_filename = 'emerging.test.annotated'
    data_dir = os.path.join('tests', 'externaldata', 'wnut2017')
    for filename in (train_filename, val_filename, test_filename):
        file_path = os.path.join(data_dir, filename)
        download(url_format.format(filename), path=file_path)
        with open(file_path) as fp:
            data = fp.readlines()
            if 'train' in filename:
                train = data
            elif 'dev' in filename:
                valid = data
            else:
                test = data

    assert len(train) == 66124, len(train)
    assert len(valid) == 16742, len(valid)
    assert len(test) == 24681, len(test)
