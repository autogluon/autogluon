
__all__ = ['Space', 'Categorical', 'Real', 'Int', 'Bool']


class Space(object):
    """Basic search space describing set of possible candidate values for hyperparameter.
    """
    @property
    def default(self):
        """Return default value of hyperparameter corresponding to this search space. This value is tried first during hyperparameter optimization."""
        raise NotImplementedError


class SimpleSpace(Space):
    def __init__(self, default):
        self._default = default

    """Non-nested search space (i.e. corresponds to a single simple hyperparameter)."""
    def __repr__(self):
        reprstr = self.__class__.__name__
        if hasattr(self, 'lower') and hasattr(self, 'upper'):
            reprstr += ': lower={}, upper={}'.format(self.lower, self.upper)
        if hasattr(self, 'value'):
            reprstr += ': value={}'.format(self.value)
        return reprstr

    @property
    def default(self):
        """Return default value of hyperparameter corresponding to this search space. This value is tried first during hyperparameter optimization."""
        return self._default

    @default.setter
    def default(self, value):
        """Set default value for hyperparameter corresponding to this search space. The default value is always tried in the first trial of HPO.
        """
        self._default = value


class Categorical(SimpleSpace):
    """Nested search space for hyperparameters which are categorical. Such a hyperparameter takes one value out of the discrete set of provided options. 
       The first value in the list of options will be the default value that gets tried first during HPO.

    Parameters
    ----------
    data : Space or python built-in objects
        the choice candidates

    Examples
    --------
    a = Categorical('a', 'b', 'c', 'd')  # 'a' will be default value tried first during HPO
    """
    def __init__(self, *data):
        self.data = [*data]
        super().__init__(self.data[0])

    def __iter__(self):
        for elem in self.data:
            yield elem

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, data):
        self.data[index] = data

    def __len__(self):
        return len(self.data)

    def convert_to_sklearn(self):
        return self.data

    def __repr__(self):
        reprstr = self.__class__.__name__ + str(self.data)
        return reprstr


class Real(SimpleSpace):
    """Search space for numeric hyperparameter that takes continuous values.

    Parameters
    ----------
    lower : float
        The lower bound of the search space (minimum possible value of hyperparameter)
    upper : float
        The upper bound of the search space (maximum possible value of hyperparameter)
    default : float (optional)
        Default value tried first during hyperparameter optimization
    log : (True/False)
        Whether to search the values on a logarithmic rather than linear scale. 
        This is useful for numeric hyperparameters (such as learning rates) whose search space spans many orders of magnitude.

    Examples
    --------
    >>> learning_rate = Real(0.01, 0.1, log=True)
    """
    def __init__(self, lower, upper, default=None, log=False):
        if log and lower <= 0:
            raise AssertionError(f'lower must be greater than 0 when `log=True`. lower: {lower}')
        if lower >= upper:
            raise AssertionError(f'lower must be less than upper. lower: {lower}, upper: {upper}')
        if default is None:
            default = lower
        super().__init__(default=default)
        self.lower = lower
        self.upper = upper
        self.log = log

    def convert_to_sklearn(self):
        from scipy.stats import loguniform, uniform

        if self.log:
            sampler = loguniform(self.lower, self.upper)
        else:
            sampler = uniform(self.lower, self.upper - self.lower)
        return sampler


class Int(SimpleSpace):
    """Search space for numeric hyperparameter that takes integer values.

    Parameters
    ----------
    lower : int
        The lower bound of the search space (minimum possible value of hyperparameter)
    upper : int
        The upper bound of the search space (maximum possible value of hyperparameter)
    default : int (optional)
        Default value tried first during hyperparameter optimization


    Examples
    --------
    >>> range = Int(0, 100)
    """
    def __init__(self, lower, upper, default=None):
        if default is None:
            default = lower
        super().__init__(default=default)
        self.lower = lower
        self.upper = upper

    def convert_to_sklearn(self):
        from scipy.stats import randint
        return randint(self.lower, self.upper+1)


class Bool(Int):
    """Search space for hyperparameter that is either True or False. 
       `ag.Bool()` serves as shorthand for: `ag.space.Categorical(True, False)`

    Examples
    --------
    pretrained = ag.space.Bool()
    """
    def __init__(self):
        super(Bool, self).__init__(0, 1)
