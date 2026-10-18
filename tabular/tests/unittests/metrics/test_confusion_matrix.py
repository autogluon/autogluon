import numpy as np
import pytest

from autogluon.tabular import TabularDataset, TabularPredictor

# The confusion matrix does not depend on model quality, only on the shape of the predictions, so
# every fit here uses the trivial model. `medium_quality` fit real models in every test.
_FIT = dict(hyperparameters={"DUMMY": {}}, fit_weighted_ensemble=False, verbosity=0)

# The two tests below need a model whose predictions actually depend on the features: a constant
# predictor cannot tell a re-transformed frame from the original, nor a refit_full model from its
# parent, so it would pass either way.
_FIT_REAL = dict(hyperparameters={"GBM": [{"num_boost_round": 10}]}, fit_weighted_ensemble=False, verbosity=0)


def test_confusion_matrix_basic(tmp_path):
    """Test basic confusion matrix computation."""
    train_data = TabularDataset(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    predictor = TabularPredictor(label="label", path=tmp_path).fit(train_data, **_FIT)

    # Compute confusion matrix without display
    cm = predictor.confusion_matrix(display=False)

    assert isinstance(cm, np.ndarray)
    assert cm.shape[0] == cm.shape[1]
    assert cm.shape[0] == 2  # Binary classification


def test_confusion_matrix_with_data(tmp_path):
    """Test confusion matrix with explicit data."""
    train_data = TabularDataset(
        {
            "feature1": [1, 2, 3, 4, 5, 6],
            "label": [0, 1, 0, 1, 0, 1],
        }
    )

    test_data = TabularDataset(
        {
            "feature1": [7, 8, 9, 10],
            "label": [0, 1, 0, 1],
        }
    )

    predictor = TabularPredictor(label="label", path=tmp_path).fit(train_data, **_FIT)

    # Compute on test data
    cm = predictor.confusion_matrix(test_data, display=False)

    assert isinstance(cm, np.ndarray)
    assert cm.shape == (2, 2)


def test_confusion_matrix_normalize(tmp_path):
    """Test normalized confusion matrix."""
    train_data = TabularDataset(
        {
            "f1": [1, 2, 3, 4, 5, 6],
            "label": [0, 1, 0, 1, 0, 1],
        }
    )

    predictor = TabularPredictor(label="label", path=tmp_path).fit(train_data, **_FIT)

    cm_norm = predictor.confusion_matrix(normalize="true", display=False)

    # Check normalization: rows should sum to 1
    assert np.allclose(cm_norm.sum(axis=1), 1.0)


def test_confusion_matrix_save_plot(tmp_path):
    """Test saving confusion matrix plot."""
    train_data = TabularDataset(
        {
            "f": [1, 2, 3, 4, 5, 6],
            "label": [0, 1, 0, 1, 0, 1],
        }
    )

    predictor = TabularPredictor(label="label", path=tmp_path).fit(train_data, **_FIT)

    save_path = tmp_path / "cm_plot.png"

    # Save without displaying
    predictor.confusion_matrix(display=False, save_path=str(save_path))

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_confusion_matrix_multiclass(tmp_path):
    """Test confusion matrix with multiclass classification."""
    train_data = TabularDataset(
        {
            "f1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "f2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
            "label": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        }
    )

    predictor = TabularPredictor(
        label="label",
        path=tmp_path,
        problem_type="multiclass",  # Force multiclass
    ).fit(train_data, **_FIT)

    cm = predictor.confusion_matrix(display=False)

    assert cm.shape == (3, 3)


def test_confusion_matrix_invalid_problem_type(tmp_path):
    """Test that confusion matrix raises error for regression."""
    train_data = TabularDataset(
        {
            "f": [1, 2, 3, 4],
            "label": [1.5, 2.5, 3.5, 4.5],  # Continuous target
        }
    )

    predictor = TabularPredictor(label="label", path=tmp_path, problem_type="regression").fit(train_data, **_FIT)

    with pytest.raises(ValueError, match="only applicable to classification"):
        predictor.confusion_matrix(display=False)


def test_confusion_matrix_missing_label(tmp_path):
    """Test error when label column is missing."""
    train_data = TabularDataset(
        {
            "f": [1, 2, 3, 4],
            "label": [0, 1, 0, 1],
        }
    )

    predictor = TabularPredictor(label="label", path=tmp_path).fit(train_data, **_FIT)

    # Data without label column
    test_data_no_label = TabularDataset({"f": [5, 6]})

    with pytest.raises(ValueError, match="must contain the target column"):
        predictor.confusion_matrix(test_data_no_label, display=False)


def test_confusion_matrix_custom_labels(tmp_path):
    """Test confusion matrix with custom label ordering."""
    # Use numeric labels to avoid type conversion issues
    train_data = TabularDataset(
        {
            "f1": [1, 2, 3, 4, 5, 6, 7, 8],
            "f2": [10, 20, 30, 40, 50, 60, 70, 80],
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    predictor = TabularPredictor(label="label", path=tmp_path, problem_type="binary").fit(train_data, **_FIT)

    # Test with reversed label order
    cm = predictor.confusion_matrix(
        labels=[1, 0],  # Reverse of default [0, 1]
        display=False,
    )

    assert cm.shape == (2, 2)
    # Verify the matrix is computed with the custom label order
    assert isinstance(cm, np.ndarray)


def test_confusion_matrix_arguments(tmp_path):
    """Test new arguments: model and decision_threshold"""
    train_data = TabularDataset(
        {
            "f1": [1, 2, 3, 4, 5, 6, 7, 8],
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    predictor = TabularPredictor(label="label", path=str(tmp_path)).fit(train_data, **_FIT)

    model_name = predictor.model_names()[0]

    # Test with specific model
    cm = predictor.confusion_matrix(model=model_name, display=False)
    assert isinstance(cm, np.ndarray)

    # Test with decision_threshold
    cm_thresh = predictor.confusion_matrix(decision_threshold=0.8, display=False)
    assert isinstance(cm_thresh, np.ndarray)


def test_confusion_matrix_bagged_oof(tmp_path):
    """Test confusion matrix using OOF prediction in bagged mode"""
    # Create slightly larger dataset to ensure we can do bagging
    train_data = TabularDataset(
        {
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "label": np.random.randint(0, 2, 100),
        }
    )

    predictor = TabularPredictor(label="label", path=str(tmp_path), problem_type="binary").fit(
        train_data, num_bag_folds=2, **_FIT
    )

    # Verify we are in bagged mode
    if predictor._trainer.bagged_mode:
        # Should use OOF since no val data passed (validation data is internal/folds)
        # And we passed no external data to confusion_matrix
        cm = predictor.confusion_matrix(display=False)
        assert isinstance(cm, np.ndarray)
        assert cm.shape == (2, 2)


def test_confusion_matrix_returns_the_matrix_when_displaying(tmp_path):
    """The matrix is returned regardless of `display`, which the docstring already promised."""
    import matplotlib

    matplotlib.use("Agg")

    train_data = TabularDataset({"feature1": [1, 2, 3, 4, 5, 6, 7, 8], "label": [0, 1] * 4})
    predictor = TabularPredictor(label="label", path=tmp_path).fit(train_data, **_FIT)

    displayed = predictor.confusion_matrix(data=train_data, display=True)
    returned = predictor.confusion_matrix(data=train_data, display=False)

    assert isinstance(displayed, np.ndarray), "`display=True` is the default and returned None"
    assert np.array_equal(displayed, returned)


def test_confusion_matrix_does_not_re_transform_internal_data(tmp_path):
    """`data=None` reads already-transformed data, so it must not be transformed again.

    Re-running the feature generator over its own output re-expands datetime columns and re-encodes
    categories, which changes the predictions -- measured at 14 of 40 rows on this kind of frame. The
    matrix would then describe predictions the model does not make.
    """
    import pandas as pd
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    rng = np.random.RandomState(0)
    n = 200
    train_data = TabularDataset(
        pd.DataFrame(
            {
                "num": rng.rand(n),
                "cat": pd.Series(np.array(["a", "b", "c"], dtype=object)[rng.randint(0, 3, n)]).astype("category"),
                "when": pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.randint(0, 400, n), unit="D"),
            }
        )
    )
    # a learnable signal: a constant predictor cannot tell the two transform paths apart
    train_data["label"] = ((train_data["num"] > 0.5) ^ (train_data["cat"] == "a")).astype(int)
    predictor = TabularPredictor(label="label", path=tmp_path).fit(
        train_data, hyperparameters={"GBM": {}}, fit_weighted_ensemble=False, verbosity=0
    )

    cm = predictor.confusion_matrix(display=False)

    X_val, y_val = predictor.load_data_internal(data="val", return_X=True, return_y=True)
    # the bug this pins is only visible when re-transforming actually changes the predictions
    assert not predictor.predict(X_val).equals(predictor.predict(X_val, transform_features=False)), (
        "this frame no longer distinguishes the two paths; the test would pass either way"
    )
    expected = sk_confusion_matrix(
        y_val,
        predictor.predict(X_val, transform_features=False),
        labels=predictor.class_labels,
    )
    assert np.array_equal(cm, expected)


def test_confusion_matrix_scores_a_refit_full_model_via_its_parent(tmp_path):
    """A refit_full model was trained on the validation rows, so scoring it on them is in-sample.

    `refit_full` also makes such a model `model_best`, so this is what `confusion_matrix()` with no
    arguments hits by default.
    """
    rng = np.random.RandomState(0)
    n = 200
    feature = rng.rand(n)
    train_data = TabularDataset(
        {
            "feature1": feature.tolist(),
            "feature2": rng.rand(n).tolist(),
            # noisy enough that fitting the validation rows is visibly easier than generalising
            "label": ((feature + rng.normal(scale=0.4, size=n)) > 0.5).astype(int).tolist(),
        }
    )
    predictor = TabularPredictor(label="label", path=tmp_path).fit(train_data, **_FIT_REAL)
    predictor.refit_full()

    best = predictor.model_best
    assert best.endswith("_FULL"), f"expected refit_full to promote a _FULL model, got {best}"
    parent = predictor._trainer.get_model_attribute(model=best, attribute="refit_full_parent")

    # the default path must match scoring the parent, not the _FULL model
    assert np.array_equal(
        predictor.confusion_matrix(display=False),
        predictor.confusion_matrix(model=parent, display=False),
    )
    # ...and the two must differ, or that assertion would hold whichever model was used
    val_data = train_data.iloc[: n // 5]
    assert not np.array_equal(
        predictor.confusion_matrix(data=val_data, model=best, display=False),
        predictor.confusion_matrix(data=val_data, model=parent, display=False),
    ), "the _FULL model and its parent predict identically here; the test has no teeth"
