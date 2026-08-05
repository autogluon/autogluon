from autogluon.tabular.models.realmlp.realmlp_model import RealMLPModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {"n_epochs": 2}


def test_realmlp():
    model_cls = RealMLPModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        verify_load_wo_cuda=True,
    )


def test_realmlp_category_codes_are_stable_across_fit_and_predict():
    """Category codes must be fixed at fit time, whatever dtypes the frames carry.

    RealMLP ordinal-encodes categories downstream, and sklearn's unknown-value check dispatches
    on the dtype of the values being transformed while calling ``np.isnan`` on the *fitted*
    categories. A column whose category dtype differs between the fit frame and a predict frame
    therefore raised ``ufunc 'isnan' not supported for the input types``. Preprocessing must
    hand the encoder identical integer codes both times.
    """
    import numpy as np
    import pandas as pd

    from autogluon.tabular.models.realmlp.realmlp_model import RealMLPModel

    rng = np.random.default_rng(0)
    n = 60
    # int-valued and object-valued categories side by side, the mix that makes the numpy view of
    # one column differ from another and from itself once a frame holds an unseen value.
    train = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "cat_int": pd.Categorical(rng.integers(1, 6, size=n)),
            "cat_str": pd.Categorical(rng.choice(list("abc"), size=n)),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n))

    model = RealMLPModel(problem_type="binary", eval_metric=None)
    model._preprocess_set_features(X=train)
    processed_train = model.preprocess(train, y=y, is_train=True, bool_to_cat=True, impute_bool=False)

    # Every category column is int-coded, so the downstream encoder sees one dtype.
    for col in model._cat_col_names:
        assert processed_train[col].cat.categories.dtype.kind in "iu", col

    # A predict frame with an unseen category and a different category dtype still maps onto the
    # fit-time codes rather than shifting them.
    predict = pd.DataFrame(
        {
            "num": rng.normal(size=5),
            "cat_int": pd.Categorical([1, 2, 3, 4, 99]),  # 99 unseen
            "cat_str": pd.Categorical(["a", "b", "c", "a", "zz"]),  # zz unseen
        }
    )
    processed_predict = model.preprocess(predict)
    for col in model._cat_col_names:
        assert processed_predict[col].cat.categories.dtype.kind in "iu", col
        unseen_code = len(model._category_mapping[col])
        assert unseen_code in set(processed_predict[col].dropna().astype(int)), col
