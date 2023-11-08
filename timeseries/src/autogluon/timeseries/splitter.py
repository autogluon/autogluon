__all__ = []


class AbstractTimeSeriesSplitter:
    def __init__(self, *args, **kwargs):
        raise ValueError(
            "`AbstractTimeSeriesSplitter` has been deprecated. "
            "Please use `autogluon.timeseries.val_splitter.ExpandingWindowSplitter` instead."
        )


class MultiWindowSplitter(AbstractTimeSeriesSplitter):
    def __init__(self, *args, **kwargs):
        raise ValueError(
            "`MultiWindowSplitter` has been deprecated. "
            "Please use `autogluon.timeseries.val_splitter.ExpandingWindowSplitter` instead."
        )


class LastWindowSplitter(MultiWindowSplitter):
    def __init__(self, *args, **kwargs):
        raise ValueError(
            "`LastWindowSplitter` has been deprecated. "
            "Please use `autogluon.timeseries.val_splitter.ExpandingWindowSplitter` instead."
        )
