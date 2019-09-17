__all__ = ['List', 'Linear', 'Log']

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


class List(object):
    """Categorical search space.

    Args:
        name: the name of the search space
        choices: the value candidates

    Example:
        >>> list_space = List('listspace', ['0', '1', '2'])
    """
    def __init__(self, name, choices):
        self.name = name
        choices = [json.loads(jsonpickle.encode(choice)) for choice in choices]
        self.choices = choices
        self.hyper_param = CSH.CategoricalHyperparameter(name=self.name,
                                                         choices=self.choices)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(self.hyper_param)
        self.search_space = cs

    def get_hyper_param(self):
        return self.hyper_param

    def __repr__(self):
        return "AutoGluon List Space %s: %s" % (self.name, str(self.choices))

    def __str__(self):
        return "AutoGluon List Space %s: %s" % (self.name, str(self.choices))


class Linear(object):
    """linear search space.

    Args:
        name: the name of the search space
        lower: the lower bound of the search space
        upper: the upper bound of the search space

    Example:
        >>> linear_space = Linear('linspace', 0, 10)
    """
    def __init__(self, name, lower, upper):
        self.name = name
        self.lower = lower
        self.upper = upper
        if isinstance(lower, int) and isinstance(upper, int):
            self.hyper_param = CSH.UniformIntegerHyperparameter(name=self.name,
                                                                lower=self.lower,
                                                                upper=self.upper,
                                                                log=False)
        else:
            self.hyper_param = CSH.UniformFloatHyperparameter(name=self.name,
                                                              lower=self.lower,
                                                              upper=self.upper,
                                                              log=False)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(self.hyper_param)
        self.search_space = cs

    def get_hyper_param(self):
        return self.hyper_param

    def __repr__(self):
        if isinstance(self.lower, int) and isinstance(self.upper, int):
            return "AutoGluon Linear Space %s: lower %d, upper %d" % (self.name,
                                                                      self.lower,
                                                                      self.upper)
        return "AutoGluon Linear Space %s: lower %f, upper %f" % (self.name,
                                                                  self.lower,
                                                                  self.upper)

    def __str__(self):
        if isinstance(self.lower, int) and isinstance(self.upper, int):
            return "AutoGluon Linear Space %s: lower %d, upper %d" % (self.name,
                                                                      self.lower,
                                                                      self.upper)
        return "AutoGluon Linear Space %s: lower %f, upper %f" % (self.name,
                                                                  self.lower,
                                                                  self.upper)


class Log(object):
    """loglinear search space.

    Args:
        name: the name of the search space
        lower: the lower bound of the search space
        upper: the upper bound of the search space

    Example:
        >>> log_space = Log('logspace', 0, 10)
    """
    def __init__(self, name, lower, upper):
        self.name = name
        self.lower = lower
        self.upper = upper
        if isinstance(lower, int) and isinstance(upper, int):
            self.hyper_param = CSH.UniformIntegerHyperparameter(name=self.name,
                                                                lower=self.lower,
                                                                upper=self.upper,
                                                                log=True)
        else:
            self.hyper_param = CSH.UniformFloatHyperparameter(name=self.name,
                                                              lower=self.lower,
                                                              upper=self.upper,
                                                              log=True)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(self.hyper_param)
        self.search_space = cs

    def get_hyper_param(self):
        return self.hyper_param


    def __repr__(self):
        if isinstance(self.lower, int) and isinstance(self.upper, int):
            return "AutoGluon Log Space %s: lower %d, upper %d" % (self.name,
                                                                      self.lower,
                                                                      self.upper)
        return "AutoGluon Log Space %s: lower %f, upper %f" % (self.name,
                                                                  self.lower,
                                                                  self.upper)

    def __str__(self):
        if isinstance(self.lower, int) and isinstance(self.upper, int):
            return "AutoGluon Log Space %s: lower %d, upper %d" % (self.name,
                                                                      self.lower,
                                                                      self.upper)
        return "AutoGluon Log Space %s: lower %f, upper %f" % (self.name,
                                                                  self.lower,
                                                                  self.upper)
