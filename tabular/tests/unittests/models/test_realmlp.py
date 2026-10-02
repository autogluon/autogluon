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


def test_realmlp_refit_full_stops_at_the_best_epoch():
    """A fit with validation data records pytabkit's best epoch in ``params_trained["stop_epoch"]``; ``refit_full``
    carries it into the refit model's hyperparameters (rounded mean over a bag's children), which trains on all rows."""
    import numpy as np
    import pandas as pd

    from autogluon.tabular import TabularPredictor

    rng = np.random.default_rng(0)
    n = 300
    train = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "c": pd.Categorical(rng.choice(list("xyz"), size=n))}
    )
    train["label"] = (train["a"] + 0.5 * train["b"] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    n_epochs = 6
    predictor = TabularPredictor(label="label", verbosity=0).fit(
        train, hyperparameters={RealMLPModel: {"n_epochs": n_epochs}}, num_bag_folds=2, fit_weighted_ensemble=False
    )
    bag_name = predictor.model_names()[0]
    bag = predictor._trainer.load_model(bag_name)
    child_epochs = [bag.load_child(child).params_trained["stop_epoch"] for child in bag.models]
    assert all(isinstance(e, int) and 1 <= e <= n_epochs for e in child_epochs), child_epochs

    refit_name = predictor.refit_full()[bag_name]
    refit = predictor._trainer.load_model(refit_name)
    refit_child = refit.load_child(refit.models[0])
    assert refit_child.params["stop_epoch"] == round(np.mean(child_epochs))
    assert not refit_child.get_info()["val_in_fit"]
    assert "stop_epoch" not in refit_child.params_trained  # no validation data, so no best epoch to record
