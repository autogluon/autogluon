class XShiftInferrer:
    """The results of running XShiftDetector.infer(), which should be run after .fit().  This will output the results
    of the detection of difference between training and test data.

    Parameters
    ----------
    detector: an XShiftDetector instance
        a fit detector for XShift

    Methods
    -------
    .json(): output the results into json format
    .print(): print the results to screen
    """
    def __init__(self, detector):
        self._detector = detector

    def json(self):
        """output the results in json format
        """
        pass

    def print(self):
        """print the results to screen
        """
        pass