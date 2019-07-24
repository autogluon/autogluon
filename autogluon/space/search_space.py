__all__ = ['Space', 'List', 'Linear', 'Log', 'Int', 'Bool', 'Constant',
           'get_config_space']

import ConfigSpace.hyperparameters as CSH

class Space(object):
    pass

class List(Space):
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


class Linear(Space):
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


class Log(Space):
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


class Int(Space):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

class Bool(IntSpace):
    def __init__(self):
        super(Bool, self).__init__(0, 1)

class Constant(Space):
    def __init__(self, val):
        self.value = val

def get_config_space(name, space):
    assert isinstance(space, Space)
    if isinstance(space, List):
        return CSH.CategoricalHyperparameter(name=name, choices=space.data)
    elif isinstance(space, Linear):
        return CSH.UniformFloatHyperparameter(name=name, lower=space.lower, upper=space.upper)
    elif isinstance(space, Log):
        return CSH.UniformFloatHyperparameter(name=name, lower=space.lower, upper=space.upper, log=True)
    elif isinstance(space, Int):
        return CSH.UniformIntegerHyperparameter(name=name, lower=space.lower, upper=space.upper)
    elif isinstance(space, Constant):
        return CSH.Constant(name=name, value=space.value)
    else:
        raise NotImplemented
