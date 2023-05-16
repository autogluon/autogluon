import joblib.externals.loky

from .naive import NaiveModel, SeasonalNaiveModel
from .statsforecast import AutoARIMAModel, AutoETSModel, DynamicOptimizedThetaModel, ThetaModel
from .statsmodels import ARIMAModel, ETSModel, ThetaStatsmodelsModel

# By default, joblib w/ loky backend kills processes that take >300MB of RAM assuming that this is caused by a memory
# leak. This leads to problems for some memory-hungry models like AutoARIMA/Theta.
# This monkey patch removes this undesired behavior
joblib.externals.loky.process_executor._MAX_MEMORY_LEAK_SIZE = int(3e10)
