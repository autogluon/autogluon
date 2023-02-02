try:
    from .version import __version__
except ImportError:  # pragma: no cover
    pass

from .state import AnalysisState
