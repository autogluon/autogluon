import pytest
from autogluon.shift import Classifier2ST
from autogluon.vision import ImageDataset, ImagePredictor
import os

data_dir = os.path.join('..','..','..','data')

def test_classifier2ST():
    twodogs_dir = os.path.join(data_dir,'two_dogs')
    data = ImageDataset.from_folder(twodogs_dir)
    pred = ImagePredictor()
    tst = Classifier2ST(pred)
    tst.fit(data, sample_label='label')
