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

    infer: outputs the results of XShift detection

    Usage
    -----
    >>> pred = TabularPredictor(label='class') #class is the Y variable
    >>> xshiftd = XShiftDetector(pred)
    >>> xshiftd.fit(train.drop(columns = 'class'), test.drop(columns = 'class'))
    >>> res = xshiftd.infer()
    >>> res.print()
    """

    def __init__(self, predictor, method="C2ST"):
        valid_methods = [
            "C2ST"
        ]
        assert method in valid_methods, f"method {method} is not one of " + ", ".join(valid_methods)
        ## Initialize classifier by checking Class of predictor?  So if tabular then you know to use tabular predictor
        ## for classifier in C2ST
        pass

    def fit(self, data):
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

    def infer(self):
        """Return the results of the XShift detection

        Returns
        -------
        XShiftInferer instance which will have methods
            - .json(): output json object of the results
            - .print(): print the results in a nice format
        """
        assert self._is_fit, "Need to call .fit() before inference"
        inferrer = sft.XShiftInferrer(self)
        return inferrer