import pytest

from autogluon.tabular.models.tabpfnv2.tabpfn3_model import TabPFN3Model
from autogluon.tabular.testing import FitHelper

toy_model_params = {"n_estimators": 1}


@pytest.mark.skip(
    reason="TabPFN-3 model weights are not available publicly without accepting a license agreement; "
    "run manually on a machine with the checkpoints in the tabpfn cache."
)
def test_tabpfn3():
    model_cls = TabPFN3Model
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        verify_load_wo_cuda=True,
        # TabPFN returns different predictions when predicting on an individual sample
        verify_single_prediction_equivalent_to_multi=False,
    )


def test_tabpfn3_and_2_6_have_no_feature_cap():
    """TabPFN-2.6 and -3 must not cap features, while the 2.5 base still does.

    `ag.max_features` skips a fit outright, and these two models are the strongest methods on
    BeyondArena's widest tasks (up to 22k columns), so a cap would exclude them exactly where
    they win. The cap must be `None` rather than absent: `_default_auxiliary_params_extra`
    entries merge base-most class first, so an absent key would inherit the 2.5 base's cap.
    """
    from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import TabPFNModel
    from autogluon.tabular.models.tabpfnv2.tabpfnv2_6_model import TabPFNv26Model

    assert TabPFN3Model()._get_default_auxiliary_params()["max_features"] is None
    assert TabPFNv26Model()._get_default_auxiliary_params()["max_features"] is None
    # The 2.5 base is unchanged, which is what makes the explicit None load-bearing.
    assert TabPFNModel()._get_default_auxiliary_params()["max_features"] == 2000


def test_tabpfn_preprocess_preserves_missing_categoricals():
    """Categorical columns reach TabPFN as `category` dtype, with missing values intact.

    Label-encoding them first loses the missingness: `.cat.codes` maps missing to -1, and
    TabPFN casts every column named in `categorical_features_indices` back to `category`
    (`tabpfn.preprocessing.clean.fix_dtypes`), which turns that -1 into an ordinary level. The
    model then sees no missing values in those columns. Applies to every TabPFN version, since
    they share this `_preprocess`.
    """
    import numpy as np
    import pandas as pd

    from autogluon.tabular.models.tabpfnv2.tabpfnv2_6_model import TabPFNv26Model

    for model_cls in (TabPFN3Model, TabPFNv26Model):
        levels = np.array(["absent", "mild", "normal"], dtype=object)
        rng = np.random.default_rng(0)
        values = levels[rng.integers(0, 3, 40)]
        values[[3, 11, 27]] = None
        X = pd.DataFrame({"num": rng.normal(size=40), "cat": pd.Series(values, dtype="category")})

        model = model_cls(problem_type="binary", eval_metric=None)
        model._preprocess_set_features(X=X)
        processed = model.preprocess(X, is_train=True)

        assert str(processed["cat"].dtype) == "category", model_cls.__name__
        assert processed["cat"].isna().sum() == 3, model_cls.__name__
        assert -1 not in set(processed["cat"].cat.categories), model_cls.__name__
        # the indices TabPFN is told about still point at the categorical column: it infers
        # modality from dtype, and an integer-levelled `category` column reads as numeric, so
        # without them such a column would be treated as NUMERICAL.
        assert model._cat_indices == [X.columns.get_loc("cat")], model_cls.__name__
        # untouched numeric column
        assert processed["num"].equals(X["num"]), model_cls.__name__
