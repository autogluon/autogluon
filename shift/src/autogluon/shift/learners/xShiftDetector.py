## This is the public API that should be exposed to the general user

import autogluon.shift as sft
from sklearn.metrics import balanced_accuracy_score
import warnings

class XShiftDetector:
    """Detect a change in covariate (X) distribution between training and test, which we call XShift.  This should be
    used after the predictor is instantiated, but before the predictor is fit.  It can tell you if your training set is
    not representative of your test set distribution.  It does this using a Classifier 2-sample test (C2ST).

    Parameters
    ----------
    PredictorClass: an AutoGluon predictor, such as instance of autogluon.tabular.Predictor
        The predictor that will be fit on training set and predict the test set

    label: str
        the Y variable that is to be predicted (if it appears in the train/test data then it will be removed)

    classification_metric: str
        the metric used for the C2ST, it must be one of ['accuracy']

    Methods
    -------
    fit: fits the detector on training and test covariate data

    json, summary: outputs the results of XShift detection
    - test statistic
    - pvalue (optional, if compute_pvalue=True in .fit)
    - detector feature importances
    - top k anomalous samples

    Usage
    -----
    >>> xshiftd = XShiftDetector(TabularPredictor, label='class')
    Fit the detector...
    >>> xshiftd.fit(Xtrain, Xtest)
    Output the decision...
    >>> xshiftd.decision()
    Output the summary...
    >>> xshiftd.summary()
    """

    def __init__(self,
                 PredictorClass,
                 label=None,
                 classification_metric = 'balanced accuracy'):
        named_metrics = {
            'balanced accuracy' : balanced_accuracy_score,
        }
        assert classification_metric in named_metrics.keys(), \
            'classification_metric must be one of [' + ', '.join(named_metrics.keys()) + ']'
        pred = PredictorClass(label='xshift_label')
        self.C2ST = sft.Classifier2ST(pred)
        self.cmetric = named_metrics[classification_metric]
        self.cmetric_name = classification_metric
        if not label:
            warnings.warn('label is not specified, please ensure that Xtrain, Xtest do not have the Y (label) '
                          'variable')
        self.label = label
        self._is_fit = False

    def _post_fit(func):
        """decorator for post-fit methods"""
        def pff_wrapper(self, *args, **kwargs):
            assert self._is_fit, f'.fit needs to be called prior to .{func.__name__}'
            return func(self, *args, **kwargs)
        return pff_wrapper

    def fit(self, Xtrain, Xtest):
        """Fit the XShift detector.

        Parameters
        ----------
        Xtrain, Xtest : pd.DataFrame
            training dataframe and test dataframe
        """
        assert 'xshift_label' not in Xtrain.columns, 'your data columns contain "xshift_label" which is used internally'

        if self.label:
            Xtrain = Xtrain.drop(columns=[self.label])
            Xtest = Xtest.drop(columns=[self.label])

        self.C2ST.fit((Xtrain, Xtest), sample_label='xshift_label', accuracy_metric=self.cmetric)

        ## Sample anomalies
        as_top = self.C2ST.sample_anomaly_scores(how='top')
        as_top = as_top[[1]].rename(columns={1: 'xshift_test_proba'})
        self.anomalies = as_top.join(Xtest)

        ## Feature importance
        if self.C2ST.has_fi:
            self.fi_scores = self.C2ST.feature_importance()
        else:
            self.fi_scores = None

        self._is_fit = True

    @_post_fit
    def decision(self, teststat_thresh = 0.55):
        """Decision function for testing XShift.  Uncertainty quantification is currently not supported.

        Parameters
        ----------
        teststat_thresh: float
            the threshold for the test statistic

        Returns
        -------
        One of ['detected', 'not detected']
        """
        ## default teststat_thresh by metric
        self.teststat_thresh = teststat_thresh
        if self.C2ST.test_stat > teststat_thresh:
            return 'detected'
        else:
            return 'not detected'

    @_post_fit
    def json(self):
        """output the results in json format
        """
        return {
            'detection status': self.decision(),
            'test statistic': self.C2ST.test_stat,
            'feature importance': self.fi_scores,
            'sample anomalies': self.anomalies
        }

    @_post_fit
    def summary(self, format = "markdown"):
        """print the results to screen
        """
        assert format == 'markdown', 'Only markdown format is supported'
        if self.decision() == 'not detected':
            ret_md = (
                f"# Detecting distribution shift"
                f"We did not detect a substantial difference between the training and test X distributions."
                )
        else:
            ret_md = (
                f"# Detecting distribution shift\n"
                f"We detected a substantial difference between the training and test X distributions,\n" 
                f"a type of distribution shift.\n"
                f"\n"
                f"## Test results\n"
                f"We can predict whether a sample is in the test vs. training set with a {self.cmetric_name} of\n"
                f"{self.C2ST.test_stat} (larger than the threshold of {self.teststat_thresh}).\n"
                f"\n"
                f"## Feature importances\n"
                f"The variables that are the most responsible for this shift are those with high feature importance:\n"
                f"{self.fi_scores.to_markdown()}"
            )
        return ret_md