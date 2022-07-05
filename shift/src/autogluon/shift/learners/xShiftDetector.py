## This is the public API that should be exposed to the general user

import shift as sft

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
    Print the results...
    >>> xshiftd.print()
    """

    def __init__(self, predictor = None, predictor_class = None, method="C2ST", metric = 'balanced accuracy'):
        valid_methods = [
            "C2ST"
        ]
        assert method in valid_methods, f"method {method} is not one of " + ", ".join(valid_methods)
        ## Initialize classifier by checking Class of predictor?  So if tabular then you know to use tabular predictor
        ## for classifier in C2ST
        pass

    def fit(self, data, compute_pvalue = False):
        """Fit the XShift detector.

        Parameters
        ----------
        data : pd.DataFrame, or tuple
            either
            - a dataframe with a label column where 1 = target and 0 = source
            - a tuple of training dataframe and test dataframe
        """
        self._is_fit = True
        pass

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
        pass

    def json(self):
        """output the results in json format
        """
        pass

    def print(self):
        """print the results to screen
        """
        pass