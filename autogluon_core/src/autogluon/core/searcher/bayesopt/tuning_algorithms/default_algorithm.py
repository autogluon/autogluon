from .bo_algorithm_components import LBFGSOptimizeAcquisition
from ..models.nphead_acqfunc import EIAcquisitionFunction

DEFAULT_ACQUISITION_FUNCTION = EIAcquisitionFunction
DEFAULT_LOCAL_OPTIMIZER_CLASS = LBFGSOptimizeAcquisition
DEFAULT_NUM_INITIAL_CANDIDATES = 250
DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS = 3
DEFAULT_METRIC = 'active_metric'


def dictionarize_objective(x):
    return {DEFAULT_METRIC: x}
