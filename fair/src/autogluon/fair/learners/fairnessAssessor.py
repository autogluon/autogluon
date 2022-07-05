class FairnessAssessor:
    """Assess the fairness of a predictor post-fit by computing protected group specific metrics.

    Parameters
    ----------
    predictor: an AutoGluon predictor
        the predictor that we will be evaluating for fairness, it should have already been fit to
        training data

    Methods
    -------
    fit: compute group-specific metrics for a predictor and aggregate

    eval: compute group-wise or aggregated metric

    fairness: compute default metrics for prediction task

    Usage
    -----
    >>> pred = TabularPredictor(label='class') #class is the Y variable
    >>> pred.fit(train_data)
    >>> fass = FairnessAssessor(pred)
    >>> fass.fit(validation_data, 'race')
    >>> res = fass.eval('accuracy', agg='min_median_ratio')
    >>> res_full = fass.fairness()
    """

    def __init__(self, predictor):
        self._predictor = predictor

    def fit(self,
            validation_data,
            protected_attribute):
        """Make predictions and prepare to evaluate metrics.

        Parameters
        ----------
        validation_data: pd.DataFrame
            validation data (what would be used in predict())

        protected_attribute: str or pd.Series
            the protected attribute (categorical) either as
            - a column name in validation_data (str)
            - or a categorical variable with same index as validation_data (pd.Series)
        """
        pass

    def eval(self,
             metric,
             agg=None):
        """Return the results of fairness assessor.

        metric: str
            the metric to be computed by protected group
            if the fairness metric does not conform to agg(per-group metric) then agg = None will
            return the single float fairness metric (e.g. equalized odds)

        agg: str
            aggregation method for combining a metric for each protected group into a single metric,
            if None then no aggregation is used and the results are by protected group, unless metric does not support
            aggregation

        Returns
        -------
        pd.Series (indexed by protected group) if protected group, float otherwise
        """
        pass

    def fairness(self):
        """Return default list of fairness metrics for prediction task

        Returns
        -------
        fairness_metrics: dict
        """
        pass
