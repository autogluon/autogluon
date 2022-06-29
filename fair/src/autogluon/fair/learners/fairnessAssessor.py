class fairnessAssessor:
    """Assess the fairness of a predictor post-fit by computing protected group specific metrics.

    Parameters
    ----------
    predictor: an AutoGluon predictor
        the predictor that we will be evaluating for fairness

    Methods
    -------
    fit: compute group-specific metrics for a predictor and aggregate

    eval: compute group-wise or aggregated metric

    Usage
    -----
    >>> pred = TabularPredictor(label='class') #class is the Y variable
    >>> fass = FairnessAssessor(pred)
    >>> fass.fit(test, 'race')
    >>> res = fass.eval('accuracy', agg='min_median_ratio')
    """

    def __init__(self, predictor):
        self._predictor = predictor

    def fit(self,
            test_data,
            protected_attribute):
        """Make predictions and prepare to evaluate metrics.

        Parameters
        ----------
        test_data: pd.DataFrame
            test data (what would be used in predict())

        protected_attribute: str or pd.Series
            the protected attribute (categorical) either as
            - a column name in test_data (str)
            - or a categorical variable with same index as test_data (pd.Series)
        """
        pass

    def eval(self,
             metric,
             agg=None):
        """Return the results of fairness assessor.

        metric: str
            the metric to be computed by protected group

        agg: str
            aggregation method for combining a metric for each protected group into a single metric,
            if None then no aggregation is used and the results are by protected group

        Returns
        -------
        pd.Series (indexed by protected group) if agg==None, float otherwise
        """
        pass
