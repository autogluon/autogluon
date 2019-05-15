__all__ = ['List', 'Linear', 'Log']

import ConfigSpace.hyperparameters as CSH


# TODO (cgraywang): add constant parameter?

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
        return "AutoGluon Linear Space %s: lower %d, upper %d" % (self.name,
                                                                  self.lower,
                                                                  self.upper)

    def __str__(self):
        return "AutoGluon Linear Space %s: lower %d, upper %d" % (self.name,
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
        return "AutoGluon Log Space %s: lower %d, upper %d" % (self.name,
                                                               self.lower,
                                                               self.upper)

    def __str__(self):
        return "AutoGluon Log Space %s: lower %d, upper %d" % (self.name,
                                                               self.lower,
                                                               self.upper)
