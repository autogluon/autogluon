try:
    from .version import __version__
except ImportError:
    pass

from .state import AnalysisState
