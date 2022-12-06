"""Tests for FairPredictor"""
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core import metrics
from autogluon import fair
from autogluon.fair.learners.fair import FairPredictor
from autogluon.fair.utils import group_metrics as gm


def test_metrics():
    "check that core.metrics give the same answer as group metrics"
    array1 = np.random.randint(0, 2, 100)
    array2 = np.random.randint(0, 2, 100)
    array3 = np.zeros(100)
    met_list = (metrics.accuracy,
                metrics.balanced_accuracy,
                metrics.f1,
                metrics.mcc,
                metrics.precision,
                metrics.recall)
    group_met_list = (
        gm.accuracy,
        gm.balanced_accuracy,
        gm.f1,
        gm.mcc,
        gm.precision,
        gm.recall)
    for met, group_met in zip(met_list, group_met_list):
        assert np.isclose(met(array1, array2), group_met(array1, array2, array3)[0], 1e-5)


def test_metrics_identities():
    """ sanity check, make sure metrics are consistent with standard identities.
     This combined with test metrics gives coverage of everything up to the clarify metrics"""
    array1 = np.random.randint(0, 2, 100)
    array2 = np.random.randint(0, 2, 100)
    array3 = np.random.randint(0, 4, 100)
    assert np.isclose(gm.pos_data_rate(array1, array2, array3),
                      1 - gm.neg_data_rate(array1, array2, array3)).all()
    assert np.isclose(gm.pos_pred_rate(array1, array2, array3),
                      1 - gm.neg_pred_rate(array1, array2, array3)).all()
    assert np.isclose(gm.true_pos_rate(array1, array2, array3),
                      1 - gm.false_neg_rate(array1, array2, array3)).all()
    assert np.isclose(gm.true_neg_rate(array1, array2, array3),
                      1 - gm.false_pos_rate(array1, array2, array3)).all()
    accuracy = gm.Utility([1, 0, 0, 1], 'accuracy')
    assert np.isclose(gm.accuracy(array1, array2, array3), accuracy(array1, array2, array3)).all()
    # assert np.isclose(gm.(A,B,array3),1-gm.(A,B,array3)).all()
    # check that additive_metrics can be called.
    assert np.isclose(gm.equalized_odds(array1, array2, array3),
                      (gm.true_pos_rate.diff(array1, array2, array3)
                       + gm.true_neg_rate.diff(array1, array2, array3)) / 2).all()


def test_fairness():
    "range of fairness tests"
    train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')[::500]
    test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
    predictor = TabularPredictor(label='class').fit(train_data=train_data)
    base_functionality(predictor, train_data)
    no_groups(predictor, test_data)
    for use_fast in [True, False]:  # use_fast=False makes this significantly slow
        predict(predictor, test_data, use_fast)
        recall_diff(predictor, test_data, use_fast)
        new_test = test_data[~test_data['race'].isin([' Other', ' Asian-Pac-Islander', ])]  # drop other
        subset(predictor, test_data, new_test, use_fast)
        disp_impact(predictor, new_test, use_fast)
        min_recall(predictor, new_test, use_fast)
        pathologoical2(predictor, new_test, use_fast)
    pathologoical(predictor, test_data)


def base_functionality(predictor, train_data):
    "not calling fit should not alter predict or predict_proba"
    fpredictor = FairPredictor(predictor, train_data, 'sex')
    fpredictor.evaluate()
    fpredictor.evaluate_fairness()
    fpredictor.evaluate_groups()
    assert (fpredictor.predict_proba(train_data) == predictor.predict_proba(train_data)).all().all()
    assert (fpredictor.predict(train_data) == predictor.predict(train_data)).all().all()
    fpredictor.evaluate(verbose=True)
    fpredictor.evaluate_fairness(verbose=True)
    fpredictor.evaluate_groups(verbose=True)
    fpredictor.evaluate_groups(verbose=True, return_original=True)
    fpredictor.evaluate_groups(return_original=True)


def no_groups(predictor, train_data):
    "check pathway works with no groups"
    fairp = fair.FairPredictor(predictor, train_data)
    fairp.evaluate()
    fairp.evaluate_groups()
    fairp.evaluate_fairness()
    assert (fairp.predict_proba(train_data) == predictor.predict_proba(train_data)).all().all()
    assert (fairp.predict(train_data) == predictor.predict(train_data)).all().all()
    fairp.fit(gm.accuracy, gm.f1, 0)


def predict(predictor, test_data, use_fast):
    "check that fairpredictor returns the same as a standard predictor before fit is called"
    fpredictor = fair.FairPredictor(predictor, test_data, groups='sex', use_fast=use_fast)
    assert all(predictor.predict(test_data) == fpredictor.predict(test_data))
    assert all(predictor.predict_proba(test_data) == fpredictor.predict_proba(test_data))


def pathologoical(predictor, train_data):
    "Returns a single constant classifier"
    fpredictor = fair.FairPredictor(predictor, train_data, groups='sex', use_fast=False)
    fpredictor.fit(metrics.roc_auc, gm.equalized_odds, 0.75)
    fpredictor.plot_frontier()
    fpredictor.evaluate_fairness()


def pathologoical2(predictor, train_data, use_fast):
    "pass it the same objective twice"
    fpredictor = fair.FairPredictor(predictor, train_data, groups='sex', use_fast=False)
    fpredictor.fit(gm.balanced_accuracy, gm.balanced_accuracy, 0)
    fpredictor.plot_frontier()
    fpredictor.evaluate_fairness()


def recall_diff(predictor, test_data, use_fast):
    """ Maximize accuracy while enforcing weak equalized odds,
    such that the difference in recall between groups is less than 2.5%"""

    fpredictor = fair.FairPredictor(predictor, test_data, 'sex', use_fast=use_fast)

    fpredictor.fit(gm.accuracy, gm.recall.diff, 0.025)

    # Evaluate the change in fairness (recall difference corresponds to EO)
    measures = fpredictor.evaluate_fairness()

    assert measures['original']['recall.diff'] > 0.025

    assert measures['updated']['recall.diff'] < 0.025


def subset(predictor, test_data, new_test, use_fast):
    "set up new fair class using 'race' as the protected group and evaluate on test data"
    fpredictor = fair.FairPredictor(predictor, test_data, 'race', use_fast=use_fast)

    full_group_metrics = fpredictor.evaluate_groups()
    fpredictor = fair.FairPredictor(predictor, new_test, 'race', use_fast=use_fast)
    partial_group_metrics = fpredictor.evaluate_groups()

    # Check that metrics computed over a subset of the data is consistent with metrics over all data
    for group in (' White', ' Black', ' Amer-Indian-Eskimo'):
        assert all(full_group_metrics.loc[group] == partial_group_metrics.loc[group])

    assert all(full_group_metrics.loc['Maximum difference'] >= partial_group_metrics.loc['Maximum difference'])


def disp_impact(predictor, new_test, use_fast):
    "Enforce the 4/5 rule that the max ratio between the proportion of positive decisions is less than 0.8"
    fpredictor = fair.FairPredictor(predictor, new_test, 'race', use_fast=use_fast)
    fpredictor.fit(gm.accuracy, gm.disparate_impact, 0.8)

    measures = fpredictor.evaluate_fairness()

    assert measures['original']['disparate_impact'] < 0.8

    assert measures['updated']['disparate_impact'] > 0.8


def min_recall(predictor, new_test, use_fast):
    "check that we can force recall >0.5 for all groups"
    fpredictor = fair.FairPredictor(predictor, new_test, 'race', use_fast=use_fast)
    # Enforce that every group has a recall over 0.5
    fpredictor.fit(gm.accuracy, gm.recall.min, 0.5)
    scores = fpredictor.evaluate_groups()
    assert all(scores['recall'][:-1] > 0.5)


def test_recall_diff_inferred():
    "use infered attributes instead of provided attributes"
    train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')[::500]
    test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
    # train two new classifiers one to predict class without using sex and one to fpredict sex without using class
    predictor, protected = fair.learners.inferred_attribute_builder(train_data, 'class', 'sex')

    for use_fast in [True, False]:
        # Build fair object using this and evaluate fairness n.b. classifier
        # accuracy decreases due to lack of access to the protected attribute, but
        # otherwise code is doing the same thing
        fpredictor = fair.FairPredictor(predictor, train_data, 'sex', inferred_groups=protected, use_fast=use_fast)

        # Enforce that the new classifier will satisfy equalised odds (recall
        # difference between protected attributes of less than 2.5%) despite not
        # using sex at run-time

        fpredictor.fit(gm.accuracy, gm.recall.diff, 0.025)

        measures = fpredictor.evaluate_fairness()

        assert measures['original']['recall.diff'] > 0.0025

        assert measures['updated']['recall.diff'] < 0.0025

        # Prove that sex isn't being used by dropping it and reevaluating.

        new_data = test_data.drop('sex', axis=1, inplace=False)
        fpredictor.evaluate_groups(new_data, test_data['sex'])
        # No test needed, code just has to run with sex dropped
