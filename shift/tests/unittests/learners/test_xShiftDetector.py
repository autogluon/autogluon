import pandas as pd
from autogluon.shift import Classifier2ST, XShiftDetector, XShiftInferrer
from autogluon.vision import ImageDataset, ImagePredictor
from autogluon.tabular import TabularPredictor
import os
import numpy as np
import math

data_dir = os.path.join('..','..','data')

def load_adult_data():
    adult_data_dir = os.path.join(data_dir,'AdultIncomeBinaryClassification')
    train_data = os.path.join(adult_data_dir, 'train_data.csv')
    test_data = os.path.join(adult_data_dir, 'test_data.csv')
    train = pd.read_csv(train_data, index_col=0)
    test = pd.read_csv(test_data, index_col=0)
    data = (train, test)
    return data

def load_adult_cs_data():
    adult_data_dir = os.path.join(data_dir,'adult_cs')
    train_data = os.path.join(adult_data_dir, 'adult_cs_train.csv')
    test_data = os.path.join(adult_data_dir, 'adult_cs_test.csv')
    train = pd.read_csv(train_data, index_col=0)
    test = pd.read_csv(test_data, index_col=0)
    data = (train, test)
    return data

def test_xsd_cs():
    train, test = load_adult_cs_data()
    xsd = XShiftDetector(TabularPredictor)
    xsd.fit(train, test, label='class')
    sumry = xsd.summary()
    js = xsd.json()
    assert xsd.decision() == 'detected'
    pass

def test_xsd():
    train, test = load_adult_data()
    xsd = XShiftDetector(TabularPredictor)
    xsd.fit(train, test, label='class')
    sumry = xsd.summary()
    js = xsd.json()
    assert xsd.decision() == "not detected"
    pass