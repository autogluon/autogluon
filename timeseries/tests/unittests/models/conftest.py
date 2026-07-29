import pytest

from autogluon.timeseries.models import Chronos2Model, ChronosModel, PerStepTabularModel, Toto2Model, TotoModel

from .common import (
    ALL_LOCAL_MODELS,
    CHRONOS2_MODEL_PATH,
    CHRONOS_BOLT_MODEL_PATH,
    CHRONOS_CLASSIC_MODEL_PATH,
    GLUONTS_MODELS,
    GLUONTS_MODELS_WITH_KNOWN_COVARIATES,
    GLUONTS_MODELS_WITH_STATIC_FEATURES,
    GLUONTS_MODELS_WITH_STATIC_FEATURES_AND_KNOWN_COVARIATES,
    INTERMITTENT_LOCAL_MODELS,
    MLFORECAST_MODELS,
    PER_STEP_TABULAR_MODELS,
    SEASONAL_LOCAL_MODELS,
    SEASONAL_LOCAL_MODELS_EXTRA,
    TOTO2_MODEL_PATH,
    get_multi_window_deepar,
    is_toto2_available,
    patch_constructor,
)


@pytest.fixture(params=ALL_LOCAL_MODELS)
def local_model_class(request):
    yield patch_constructor(request.param)


@pytest.fixture(params=SEASONAL_LOCAL_MODELS + SEASONAL_LOCAL_MODELS_EXTRA)
def seasonal_local_model_class(request):
    yield patch_constructor(request.param)


@pytest.fixture(params=INTERMITTENT_LOCAL_MODELS)
def intermittent_local_model_class(request):
    yield patch_constructor(request.param)


@pytest.fixture(params=GLUONTS_MODELS)
def gluonts_model_class(request):
    yield patch_constructor(request.param)


@pytest.fixture(params=GLUONTS_MODELS_WITH_STATIC_FEATURES)
def gluonts_model_with_static_features_class(request):
    yield patch_constructor(request.param)


@pytest.fixture(params=GLUONTS_MODELS_WITH_KNOWN_COVARIATES)
def gluonts_model_with_known_covariates_class(request):
    yield patch_constructor(request.param)


@pytest.fixture(params=GLUONTS_MODELS_WITH_STATIC_FEATURES_AND_KNOWN_COVARIATES)
def gluonts_model_with_known_covariates_and_static_features_class(request):
    yield patch_constructor(request.param)


@pytest.fixture(params=MLFORECAST_MODELS)
def mlforecast_model_class(request):
    yield patch_constructor(request.param)


@pytest.fixture()
def per_step_tabular_model_class():
    yield PerStepTabularModel


@pytest.fixture()
def multi_window_deepar_model_class():
    yield get_multi_window_deepar


@pytest.fixture(params=[CHRONOS_BOLT_MODEL_PATH, CHRONOS_CLASSIC_MODEL_PATH])
def chronos_zero_shot_model_class(request):
    yield patch_constructor(ChronosModel, extra_hyperparameters={"model_path": request.param})


@pytest.fixture(
    params=[  # model_path, fine_tune
        (CHRONOS_BOLT_MODEL_PATH, False),
        (CHRONOS_CLASSIC_MODEL_PATH, False),
        (CHRONOS_BOLT_MODEL_PATH, True),
        (CHRONOS_CLASSIC_MODEL_PATH, True),
    ]
)
def chronos_model_class(request):
    extra_hyperparameters = {"model_path": request.param[0]}
    if request.param[1]:
        extra_hyperparameters |= {"fine_tune": True, "fine_tune_steps": 10}

    yield patch_constructor(ChronosModel, extra_hyperparameters=extra_hyperparameters)


def patch_toto_constructor():
    """Return TotoModel constructor with MockTotoForecaster applied."""
    from .test_toto import MockTotoForecaster, noop

    def toto_model(*args, **kwargs):
        model = TotoModel(*args, **kwargs)
        model.load_forecaster = noop
        model._forecaster = MockTotoForecaster()  # type: ignore
        return model

    return toto_model


def toto2_model_param():
    """Return the real tiny Toto 2.0 model constructor, skipped when ``toto-2`` is not installed."""
    return pytest.param(
        patch_constructor(Toto2Model, extra_hyperparameters={"model_path": TOTO2_MODEL_PATH, "device": "cpu"}),
        marks=pytest.mark.skipif(not is_toto2_available(), reason="`toto-2` package is not installed"),
    )


@pytest.fixture(
    scope="session",
    params=(
        GLUONTS_MODELS
        + SEASONAL_LOCAL_MODELS
        + INTERMITTENT_LOCAL_MODELS
        + MLFORECAST_MODELS
        + PER_STEP_TABULAR_MODELS
        + [
            patch_constructor(ChronosModel, extra_hyperparameters={"model_path": CHRONOS_BOLT_MODEL_PATH}),
            patch_constructor(ChronosModel, extra_hyperparameters={"model_path": CHRONOS_CLASSIC_MODEL_PATH}),
            patch_constructor(
                ChronosModel,
                extra_hyperparameters={
                    "model_path": CHRONOS_BOLT_MODEL_PATH,
                    "fine_tune": True,
                    "fine_tune_steps": 10,
                },
            ),
            patch_constructor(
                ChronosModel,
                extra_hyperparameters={
                    "model_path": CHRONOS_CLASSIC_MODEL_PATH,
                    "fine_tune": True,
                    "fine_tune_steps": 10,
                },
            ),
            patch_constructor(Chronos2Model, extra_hyperparameters={"model_path": CHRONOS2_MODEL_PATH}),
            patch_constructor(
                Chronos2Model,
                extra_hyperparameters={
                    "model_path": CHRONOS2_MODEL_PATH,
                    "fine_tune": True,
                    "fine_tune_steps": 10,
                },
            ),
            patch_toto_constructor(),
            toto2_model_param(),
        ]
    ),
)
def model_class(request):
    yield patch_constructor(request.param)


@pytest.fixture(
    scope="session",
    params=(
        SEASONAL_LOCAL_MODELS
        + INTERMITTENT_LOCAL_MODELS
        + MLFORECAST_MODELS
        + [
            patch_constructor(ChronosModel, extra_hyperparameters={"model_path": CHRONOS_BOLT_MODEL_PATH}),
            patch_constructor(ChronosModel, extra_hyperparameters={"model_path": CHRONOS_CLASSIC_MODEL_PATH}),
            patch_constructor(Chronos2Model, extra_hyperparameters={"model_path": CHRONOS2_MODEL_PATH}),
            patch_toto_constructor(),
            toto2_model_param(),
        ]
    ),
)
def inference_only_model_class(request):
    yield patch_constructor(request.param)
