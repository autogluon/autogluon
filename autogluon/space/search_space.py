__all__ = ['List', 'Linear', 'Log', 'Exponential']
from typing import AnyStr
import ConfigSpace.hyperparameters as CSH


class List(object):
    def __init__(self, name, choices):
        self.name = name
        self.choices = choices
        self.hyper_param = CSH.CategoricalHyperparameter(name=self.name,
                                                         choices=self.choices)

    def get_hyper_param(self):
        return self.hyper_param

    def __repr__(self):
        return "AutoGluon List Space %s: %s" % (self.name, str(self.choices))

    def __str__(self):
        return "AutoGluon List Space %s: %s" % (self.name, str(self.choices))


class Linear(object):
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


class Exponential(object):
    def __init__(self, name: AnyStr, base: int, lower_exponent: int, upper_exponent: int):
        self.name = name
        self.base = base
        self.lower_exponent = lower_exponent
        self.upper_exponent = upper_exponent
        self.values = []
        for i in range(lower_exponent, upper_exponent + 1):
            self.values.append(base ** i)

        self.hyper_param = CSH.CategoricalHyperparameter(name=self.name,
                                                         choices=self.values)

    def get_hyper_param(self):
        return self.hyper_param

    def __repr__(self):
        return "AutoGluon Exponential Space %s: base %d, lower_exp %d, upper_exp %d" % (self.name,
                                                                                        self.base,
                                                                                        self.lower_exponent,
                                                                                        self.upper_exponent)

    def __str__(self):
        return "AutoGluon Exponential Space %s: base %d, lower_exp %d, upper_exp %d" % (self.name,
                                                                                        self.base,
                                                                                        self.lower_exponent,
                                                                                        self.upper_exponent)
