"""
Unit tests to ensure correctness of internal stacking logic.
"""

import os
import uuid
import shutil

from pandas.testing import assert_frame_equal, assert_series_equal

from autogluon.core.utils import generate_train_test_split_combined
from autogluon.tabular import TabularPredictor
from autogluon.tabular.testing import FitHelper


# TODO: Note that parallel fit can change predictor.model_names order depending on which model finishes fitting first, rather than by priority order.
#  We may want to find a way to make it be the same `predictor.model_names` order based on priority, or at least make input to stacker models be the same
#  This issue can theoretically lead to non-deterministic results in rare cases / edge cases
#  Might be able to fix by not using `list(model_graph.nodes)` to produce model names, but instead keeping track of ordered model names as a variable in trainer
# Note: FASTAI produces a different result when fit with parallel mode when the cpus per fold/model differ.
#  The difference is usually extremely small (seems to be numerical precision), unless it impacts the early stopping iteration.
def test_bagged_deterministic():
    """
    Tests that bagged models get a deterministic result, regardless of how they are trained

    Tests 4 configs:
    sequential fit + sequential bag
    sequential fit + parallel bag
    parallel fit + parallel bag
    parallel fit + sequential bag
    """
    sample_size = 100
    dataset_name = "adult"
    directory_prefix = "./datasets/"
    train_data, test_data, dataset_info = FitHelper.load_dataset(name=dataset_name, directory_prefix=directory_prefix)

    label = dataset_info["label"]

    init_args = dict(label=label, eval_metric="log_loss", problem_type=dataset_info["problem_type"])

    save_path = os.path.join(directory_prefix, dataset_name, f"AutogluonOutput_{uuid.uuid4()}")

    train_data, _ = generate_train_test_split_combined(
        data=train_data,
        label=init_args["label"],
        problem_type=init_args["problem_type"],
        train_size=sample_size,
    )
    test_data, _ = generate_train_test_split_combined(
        data=test_data,
        label=init_args["label"],
        problem_type=init_args["problem_type"],
        train_size=sample_size,
    )

    fit_args = dict(
        train_data=train_data,
        hyperparameters={
            "GBM": {},
            "DUMMY": {},
            # "FASTAI": {},  # Note: FASTAI is not identical between sequential and parallel fit if CPU counts differ during model fit
        },
        num_bag_folds=4,
        num_stack_levels=1,
        calibrate=True,  # ensure calibration is also deterministic
    )

    # sequential fit with sequential bag
    predictor_seq_1 = TabularPredictor(path=save_path + "seq_1", **init_args).fit(
        fit_strategy="sequential", ag_args_ensemble={"fold_fitting_strategy": "sequential_local"}, **fit_args
    )

    # sequential fit with parallel bag
    predictor_seq_2 = TabularPredictor(path=save_path + "seq_2", **init_args).fit(
        fit_strategy="sequential", **fit_args
    )

    # parallel fit with sequential bag
    predictor_par_1 = TabularPredictor(path=save_path + "par_1", **init_args).fit(
        fit_strategy="parallel", ag_args_ensemble={"fold_fitting_strategy": "sequential_local"}, **fit_args
    )

    # parallel fit with parallel bag
    predictor_par_2 = TabularPredictor(path=save_path + "par_2", **init_args).fit(fit_strategy="parallel", **fit_args)

    p = predictor_seq_1

    y_pred = p.predict(test_data)
    y_pred_proba = p.predict_proba(test_data)
    scores = p.evaluate(test_data)
    model_names = p.model_names()

    features_per_model = {}
    child_order_per_model = {}
    for model_name in model_names:
        m = p._trainer.load_model(model_name)
        features_per_model[model_name] = m.features
        child_order_per_model[model_name] = p._trainer.load_model(model_name).models

    for p2 in [predictor_seq_2, predictor_par_1, predictor_par_2]:
        y_pred_2 = p2.predict(test_data)
        y_pred_proba_2 = p2.predict_proba(test_data)
        scores_2 = p2.evaluate(test_data)
        model_names_2 = p2.model_names()

        # TODO: To truly ensure equivalence, these should be the same order.
        #  Currently this is not guaranteed because the order is based on the model fit order which is non-deterministic for parallel fit.
        # assert model_names == model_names_2
        assert sorted(model_names) == sorted(model_names_2)

        for model_name in model_names:
            m = p2._trainer.load_model(model_name)
            # To ensure equivalence, these should be the same order.
            #  Regardless of which order models finish training in parallel,
            #  they must be specified in the same order for the out-of-fold predictions to stacker models
            #  This matters for edge-cases where a model would use the column order to determine what to do.
            #  For example, during tie-breaks in the weighted ensemble.
            assert features_per_model[model_name] == m.features
            assert sorted(features_per_model[model_name]) == sorted(m.features)

            child_models = child_order_per_model[model_name]
            child_models_2 = m.models

            if not isinstance(child_models[0], str):
                child_models = [c.name for c in child_models]
                child_models_2 = [c.name for c in child_models_2]

            # This order must be equivalent, otherwise when averaging the predictions of the fold models
            #  numerical imprecision causes slight differences in results
            assert child_models == child_models_2

        assert_series_equal(y_pred, y_pred_2, check_exact=True)
        assert_frame_equal(y_pred_proba, y_pred_proba_2, check_exact=True)
        assert scores == scores_2

    for predictor in [predictor_seq_1, predictor_seq_2, predictor_par_1, predictor_par_2]:
        shutil.rmtree(predictor.path, ignore_errors=True)
