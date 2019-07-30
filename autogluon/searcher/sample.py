
__all__ = ['Sample', 'sample_from']

class Sample(object):
    pass

class sample_from(Sample):
    """Specify that autogluon should sample configuration values from
    this function.  The use of function arguments in configs must be
    disambiguated by wrapped the function in sample_from()
    Arguments:
        func: An callable function to draw a sample from.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self):
        return self.func()

    def __str__(self):
        return "autogluon.sample_from({})".format(str(self.func))

    def __repr__(self):
        return "autogluon.sample_from({})".format(repr(self.func))
