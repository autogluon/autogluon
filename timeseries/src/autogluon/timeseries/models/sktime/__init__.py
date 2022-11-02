import autogluon.timeseries as agts

if not agts.SKTIME_INSTALLED:
    raise ImportError(
        "The sktime models in autogluon.timeseries depend on sktime v0.13.1 or greater (below v0.14)."
        "Please install a suitable version of sktime in order to use these models, with "
        "`pip install 'sktime>=0.13.1,<0.14`."
    )


from .abstract_sktime import AbstractSktimeModel
from .models import ARIMASktimeModel, AutoARIMASktimeModel, AutoETSSktimeModel, TBATSSktimeModel, ThetaSktimeModel

__all__ = [
    "ARIMASktimeModel",
    "AutoARIMASktimeModel",
    "AutoETSSktimeModel",
    "TBATSSktimeModel",
    "ThetaSktimeModel",
]
