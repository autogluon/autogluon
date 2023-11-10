class TimeSeriesEvaluator:
    def __init__(self, *args, **kwargs):
        raise ValueError(
            "`TimeSeriesEvaluator` has been deprecated. "
            "Please use the metrics defined in `autogluon.timeseries.metrics` instead."
        )
