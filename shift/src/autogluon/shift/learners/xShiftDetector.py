## This is the public API that should be exposed to the general user

import autogluon.shift as sft

class XShiftDetector:
    """Detect a change in covariate (X) distribution between training and test, which we call XShift.  This should be
    used after the predictor is instantiated, but before the predictor is fit.  It can tell you if your training set is
    not representative of your test set distribution.

    Parameters
    ----------
    predictor: an AutoGluon predictor, such as instance of autogluon.tabular.Predictor
        The predictor that will be fit on training set and predict the test set

    method: str
        one of
        - "C2ST": performs classifier two sample test to detect XShift

    Methods
    -------
    fit: fits the detector on training and test covariate data

    json, print: outputs the results of XShift detection
    - test statistic
    - pvalue (optional, if compute_pvalue=True in .fit)
    - detector feature importances
    - top k anomalous samples

    Usage
    -----
    >>> pred = TabularPredictor() #class is the Y variable
    >>> xshiftd = XShiftDetector(predictor = pred)
    Alternatively, specify the predictor class...
    >>> xshiftd = XShiftDetector(predictor_class = TabularPredictor)
    Input the binary classification metric for evaluating the test set detector (if method='C2ST')...
    >>> xshiftd = XShiftDetector(predictor_class = TabularPredictor, metric = "F1")
    Fit the detector...
    >>> xshiftd.fit(Xtrain, Xtest)
    Output the decision...
    >>> xshiftd.decision()
    """

    def __init__(self,
                 PredictorClass,
                 method="C2ST",
                 metric = 'balanced accuracy'):
        valid_methods = [
            "C2ST"
        ]
        assert method in valid_methods, f"method {method} is not one of " + ", ".join(valid_methods)
        if PredictorClass:
            pred = PredictorClass(label='xshift_label')
            self.C2ST = sft.Classifier2ST(pred)
        else:
            assert False, 'One of predictor or PredictorClass must be specified'
        self.metric = metric

    def fit(self, Xtrain, Xtest, label=None, compute_pvalue = False):
        """Fit the XShift detector.

        Parameters
        ----------
        Xtrain, Xtest : pd.DataFrame
            tuple of training dataframe and test dataframe

        label: str
            the Y variable that is to be predicted (needs to be removed)

        compute_pvalue: bool
            whether to compute the p-value or not
        """
        assert 'xshift_label' not in Xtrain.columns, 'your data columns contain "xshift_label" which is used internally'
        assert compute_pvalue == False, 'compute_pvalue not supported'
        if label:
            Xtrain = Xtrain.drop(columns=[label])
            Xtest = Xtest.drop(columns=[label])

        self.C2ST.fit((Xtrain, Xtest), sample_label='xshift_label')

        ## Sample anomalies
        as_top = self.C2ST.sample_anomaly_scores(how='top')
        as_top = as_top[[1]].rename(columns={1: 'xshift_test_proba'})
        self.anomalies = as_top.join(Xtest)

        ## Feature importance
        self.fi_scores = self.C2ST.feature_importance()

        self._is_fit = True

    def decision(self, teststat_thresh = 0.55, pvalue_thresh = None):
        """Decision function for testing XShift

        Parameters
        ----------
        teststat_thresh: float
            the threshold for the test statistic

        pvalue_thresh: float
            the pvalue threshold for the permutation test

        Returns
        -------
        One of ['detected', 'not detected']
        """
        self.teststat_thresh = teststat_thresh
        if self.C2ST.test_stat > teststat_thresh: # what to do with p-value?
            return 'detected'
        else:
            return 'not detected'

    def json(self):
        """output the results in json format
        """
        return {
            'detection status': self.decision(),
            'test statistic': self.C2ST.test_stat,
            'feature importance': self.fi_scores,
            'sample anomalies': self.anomalies
        }

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
                f"We can predict whether a sample is in the test vs. training set with a {self.metric} of\n"
                f"{self.C2ST.test_stat} (larger than the threshold of {self.teststat_thresh}).\n"
                f"\n"
                f"## Feature importances\n"
                f"The variables that are the most responsible for this shift are those with high feature importance:\n"
                f"{self.fi_scores.to_markdown()}"
            )
        return ret_md