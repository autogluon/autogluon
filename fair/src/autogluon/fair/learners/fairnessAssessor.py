class fairnessAssessor:
    """Assess the fairness of a predictor post-fit by computing group specific metrics.

    Parameters
    ----------
    predictor: an AutoGluon predictor
        the predictor that we will be evaluating for fairness

    Methods
    -------
    fit: compute group-specific metrics for a predictor and aggregate

    summarize: output the results
    """

    def __init__(self, predictor):
        self._predictor = predictor

    def fit(self,
            test_data,
            protected_attribute,
            metrics='all',
            aggregations='all'):
        """Compute the group-specific metrics for the prediction based on the protected attribute.

        Parameters
        ----------
        test_data: pd.DataFrame
            test data (what would be used in predict())

        protected_attribute: str or pd.Series
            the protected attribute (categorical) either as
            - a column name in test_data (str)
            - or the column variable with same index as test_data (pd.Series)

        metrics: list or str
            metrics to compute for each protected group
            - if list then it should be a list of metrics (either by name or callable)
            - 'all' will compute all available metrics for that prediction task

        aggregations: list or str
            aggregation methods for combining a metric for each protected group into a single metric
            - if list then it should be a list of aggregation methods
            - 'all' will compute all available aggregations for that prediction task
            options: 'max_min_diff', 'max_min_ratio', 'max_abs_dev', 'max_abs_ratio'
        """
        pass

    def summarize(self, how='table'):
        """Return the results of fairness assessor.

        Returns
        -------
        how = "table": returns a table with the rows as aggregation methods and columns as metrics
        """
        pass
