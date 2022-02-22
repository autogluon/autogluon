def test_module_importable():
    from autogluon.forecasting.models.abstract import (  # noqa: F401
        AbstractForecastingModel,
    )
