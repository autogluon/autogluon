from autogluon.tabular.models.tabdpt.tabdpt_model import TabDPTModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {}


def test_tabdpt():
    model_cls = TabDPTModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        verify_load_wo_cuda=True,
        # TabDPT returns different predictions when predicting on an individual sample
        verify_single_prediction_equivalent_to_multi=False,
    )


def test_tabdpt_preprocess_preserves_missing_categoricals():
    """Missing categorical values reach TabDPT as NaN, not as `.cat.codes`' -1.

    TabDPT appends a binary missing-indicator column per affected feature and then mean-imputes
    (`tabdpt.estimator`), so a -1 both hides the missingness and puts an out-of-range value on
    the feature's numeric axis.
    """
    import numpy as np
    import pandas as pd

    from autogluon.tabular.models.tabdpt.tabdpt_model import TabDPTModel

    levels = np.array(["absent", "mild", "normal"], dtype=object)
    rng = np.random.default_rng(0)
    values = levels[rng.integers(0, 3, 30)]
    values[[2, 9, 21]] = None
    X = pd.DataFrame({"num": rng.normal(size=30), "cat": pd.Series(values, dtype="category")})

    model = TabDPTModel(problem_type="binary", eval_metric=None)
    model._preprocess_set_features(X=X)
    processed = model.preprocess(X, is_train=True)

    assert isinstance(processed, np.ndarray)
    cat_column = processed[:, list(X.columns).index("cat")]
    assert np.isnan(cat_column).sum() == 3
    assert -1 not in set(cat_column[~np.isnan(cat_column)])


def test_tabdpt_allows_more_than_ten_classes():
    """The class cap reflects the library's digit decomposition, not the head width.

    `tabdpt.classifier._predict_large_cls` encodes a label in base `max_num_classes` and sums the
    per-digit log-probabilities, so class counts above the checkpoint's output head are supported
    at the cost of one forward pass per digit.
    """
    from autogluon.tabular.models.tabdpt.tabdpt_model import TabDPTModel
    from autogluon.tabular.models.tabdpt.tabdpt_turbo_model import TabDPTTurboModel

    for model_cls in (TabDPTModel, TabDPTTurboModel):
        assert model_cls(problem_type="multiclass")._get_default_auxiliary_params()["max_classes"] == 160


def test_label_encode_categoricals_preserve_missing_flag():
    """The shared helper's flag is what keeps missingness; off by default for other callers."""
    import numpy as np
    import pandas as pd

    from autogluon.tabular.models.tabdpt.tabdpt_model import TabDPTModel

    values = np.array(["a", "b", "c"], dtype=object)[[0, 1, 2, 0, 1]].astype(object)
    values[[1, 3]] = None
    X = pd.DataFrame({"cat": pd.Series(values, dtype="category")})

    default = TabDPTModel(problem_type="binary")._label_encode_categoricals(X.copy())
    preserved = TabDPTModel(problem_type="binary")._label_encode_categoricals(X.copy(), preserve_missing=True)

    # default keeps pandas' -1-for-missing convention, which other callers rely on
    assert (default["cat"] == -1).sum() == 2
    assert default["cat"].isna().sum() == 0
    # the flag restores the missingness instead
    assert preserved["cat"].isna().sum() == 2
    assert (preserved["cat"].dropna() == -1).sum() == 0
    # the non-missing codes are untouched
    assert list(default["cat"][[0, 2, 4]]) == list(preserved["cat"][[0, 2, 4]])
