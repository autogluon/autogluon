import joblib.externals.loky

from .naive import AverageModel, NaiveModel, SeasonalAverageModel, SeasonalNaiveModel
from .npts import NPTSModel
from .statsforecast import ARIMAModel, AutoARIMAModel, AutoETSModel, DynamicOptimizedThetaModel, ETSModel, ThetaModel

# By default, joblib w/ loky backend kills processes that take >300MB of RAM assuming that this is caused by a memory
# leak. This leads to problems for some memory-hungry models like AutoARIMA/Theta.
# This monkey patch removes this undesired behavior
joblib.externals.loky.process_executor._MAX_MEMORY_LEAK_SIZE = int(3e10)
