from autogluon.common.utils.deprecated_utils import Deprecated

__all__ = []


@Deprecated(
    min_version_to_warn="1.0",
    min_version_to_error="1.0",
    new="autogluon.timeseries.val_splitter.ExpandingWindowSplitter",
)
class AbstractTimeSeriesSplitter:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


@Deprecated(
    min_version_to_warn="1.0",
    min_version_to_error="1.0",
    new="autogluon.timeseries.val_splitter.ExpandingWindowSplitter",
)
class MultiWindowSplitter(AbstractTimeSeriesSplitter):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


@Deprecated(
    min_version_to_warn="1.0",
    min_version_to_error="1.0",
    new="autogluon.timeseries.val_splitter.ExpandingWindowSplitter",
)
class LastWindowSplitter(MultiWindowSplitter):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
