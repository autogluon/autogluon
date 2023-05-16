from .naive import NaiveModel, SeasonalNaiveModel
from .statsmodels import ETSStatsmodelsModel, ARIMAStatsmodelsModel, ThetaStatsmodelsModel
from .statsforecast import AutoARIMAModel, AutoETSModel, DynamicOptimizedThetaModel, ThetaModel, ETSModel


import joblib.externals.loky

joblib.externals.loky.process_executor._MAX_MEMORY_LEAK_SIZE = int(3e10)
