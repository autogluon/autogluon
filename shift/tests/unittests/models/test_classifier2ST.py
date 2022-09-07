import pandas as pd
from autogluon.shift.models import Classifier2ST
from autogluon.vision import ImageDataset, ImagePredictor
import os
import numpy as np
import math

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(THIS_DIR,'..','..','data')

def load_adult_data():
    adult_data_dir = os.path.join(data_dir,'AdultIncomeBinaryClassification')
    train_data = os.path.join(adult_data_dir, 'train_data.csv')
    test_data = os.path.join(adult_data_dir, 'test_data.csv')
    train = pd.read_csv(train_data)
    test = pd.read_csv(test_data)
    data = (train, test)
    return data

def get_dogs_data():
    twodogs_dir = os.path.join(data_dir, 'two_dogs')
    data = ImageDataset.from_folder(twodogs_dir)
    return data

def fit_tst():
    data = get_dogs_data()
    tst = Classifier2ST(ImagePredictor, sample_label='label')
    tst.fit(data)
    return tst

def test_make_source_target_label():
    data = get_dogs_data()
    source, target = data.query('label == 0'), data.query('label == 1')
    data2 = Classifier2ST._make_source_target_label((source, target), 'shift_label')
    assert target.shape[0] == data2['shift_label'].sum()

def test_classifier2ST_fit():
    tst = fit_tst()
    val = 0.82
    assert math.isclose(tst.test_stat, val, abs_tol = 2e-1)

def test_sample_anomaly_score():
    tst = fit_tst()
    as_top = tst.sample_anomaly_scores(how='top', sample_size=100)
    assert as_top.shape[0] == 100
    idx = as_top.index[[0, -1]]
    test_s = tst._test.loc[idx]
    prob_s = tst.classifier.predict_proba(test_s)
    assert prob_s.iloc[0,1] > prob_s.iloc[1,1]
    assert as_top.shape[0] == 100

def test_sample_anomaly_score_all():
    tst = fit_tst()
    ascr = tst.sample_anomaly_scores(how='all')
    pass

def test_null_test():
    data = get_dogs_data()
    data['label'] = np.random.permutation(data['label'])
    tst = Classifier2ST(ImagePredictor, sample_label='label')
    tst.fit(data)
    val = 0.5
    assert math.isclose(tst.test_stat, val, abs_tol = 1e-1), \
        f'test stat ({tst.test_stat}) is too far from similar ({val})'
    print(f'succeeded with null stat of {tst.test_stat}')

def test_classifier2ST_pvalue():
    tst = fit_tst()
    pval = tst.pvalue(num_permutations=100)
    assert pval < 0.05
