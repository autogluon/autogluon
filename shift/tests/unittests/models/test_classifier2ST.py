from autogluon.shift import Classifier2ST
from autogluon.vision import ImageDataset, ImagePredictor
import os
import numpy as np
import math

data_dir = os.path.join('..','..','data')

def get_dogs_data():
    twodogs_dir = os.path.join(data_dir, 'two_dogs')
    data = ImageDataset.from_folder(twodogs_dir)
    return data

def fit_tst():
    data = get_dogs_data()
    pred = ImagePredictor()
    tst = Classifier2ST(pred)
    tst.fit(data, sample_label='label')
    return tst

def test_make_source_target_label():
    split = 0.5
    data = get_dogs_data()
    source, target = data.query('label == 0'), data.query('label == 1')
    data2 = Classifier2ST._make_source_target_label((source, target))
    assert True

def test_classifier2ST_fit():
    tst = fit_tst()
    val = 0.82
    assert math.isclose(tst.test_stat, val, abs_tol = 1e-1), \
        f'test stat ({tst.test_stat}) is too far from similar ({val})'

def test_null_test():
    data = get_dogs_data()
    data['label'] = np.random.permutation(data['label'])
    pred = ImagePredictor()
    tst = Classifier2ST(pred)
    tst.fit(data, sample_label='label')
    val = 0.5
    assert math.isclose(tst.test_stat, val, abs_tol = 1e-1), \
        f'test stat ({tst.test_stat}) is too far from similar ({val})'
    print(f'succeeded with null stat of {tst.test_stat}')

def test_classifier2ST_pvalue():
    tst = fit_tst()
    pval = tst.pvalue(num_permutations=100)
    assert pval < 0.05


