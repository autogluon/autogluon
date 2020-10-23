# TODO: This code is comparing HyperparameterRanges_CS with HyperparameterRanges.
# If the latter code is removed, this test can go as well.

import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from numpy.testing import assert_allclose

from autogluon.core.searcher.bayesopt.autogluon.hp_ranges import \
    HyperparameterRanges_CS
from autogluon.core.searcher.bayesopt.datatypes.hp_ranges import \
    HyperparameterRanges_Impl, HyperparameterRangeCategorical, \
    HyperparameterRangeContinuous, HyperparameterRangeInteger
from autogluon.core.searcher.bayesopt.datatypes.scaling import \
    LinearScaling, LogScaling


def test_to_ndarray():
    np.random.seed(123456)
    random_state = np.random.RandomState(123456)
    prob_categ = 0.3

    for iter in range(20):
        # Create ConfigurationSpace
        num_hps = np.random.randint(low=1, high=20)
        if iter == 0:
            _prob_categ = 0.
        elif iter == 1:
            _prob_categ = 1.
        else:
            _prob_categ = prob_categ
        config_space = CS.ConfigurationSpace()
        ndarray_size = 0
        _hp_ranges = dict()
        for hp_it in range(num_hps):
            name = str(hp_it)
            if np.random.random() < _prob_categ:
                num_choices = np.random.randint(low=2, high=11)
                choices = tuple([str(i) for i in range(num_choices)])
                hp = CSH.CategoricalHyperparameter(name, choices=choices)
                hp2 = HyperparameterRangeCategorical(name, choices)
                ndarray_size += num_choices
            else:
                ndarray_size += 1
                rand_coin = np.random.random()
                if rand_coin < 0.5:
                    log_scaling = (rand_coin < 0.25)
                    hp = CSH.UniformFloatHyperparameter(
                        name=name, lower=0.5, upper=5., log=log_scaling)
                    hp2 = HyperparameterRangeContinuous(
                        name, lower_bound=0.5, upper_bound=5.,
                        scaling=LogScaling() if log_scaling else LinearScaling())
                else:
                    log_scaling = (rand_coin < 0.75)
                    hp = CSH.UniformIntegerHyperparameter(
                        name=name, lower=2, upper=10, log=log_scaling)
                    hp2 = HyperparameterRangeInteger(
                        name=name, lower_bound=2, upper_bound=10,
                        scaling=LogScaling() if log_scaling else LinearScaling())
            config_space.add_hyperparameter(hp)
            _hp_ranges[name] = hp2
        hp_ranges_cs = HyperparameterRanges_CS(config_space)
        hp_ranges = HyperparameterRanges_Impl(
            *[_hp_ranges[x] for x in config_space.get_hyperparameter_names()])
        # Compare ndarrays created by both codes
        for cmp_it in range(5):
            config_cs = hp_ranges_cs.random_candidate(random_state)
            _config = config_cs.get_dictionary()
            config = (_config[name]
                      for name in config_space.get_hyperparameter_names())
            ndarr_cs = hp_ranges_cs.to_ndarray(config_cs)
            ndarr = hp_ranges.to_ndarray(config)
            assert_allclose(ndarr_cs, ndarr, rtol=1e-4)


def test_to_ndarray_name_last_pos():
    np.random.seed(123456)
    random_state = np.random.RandomState(123456)

    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameters([
        CSH.UniformFloatHyperparameter('a', lower=0., upper=1.),
        CSH.UniformIntegerHyperparameter('b', lower=2, upper=3),
        CSH.CategoricalHyperparameter('c', choices=('1', '2', '3')),
        CSH.UniformIntegerHyperparameter('d', lower=2, upper=3),
        CSH.CategoricalHyperparameter('e', choices=('1', '2'))])
    hp_a = HyperparameterRangeContinuous(
        'a', lower_bound=0., upper_bound=1., scaling=LinearScaling())
    hp_b = HyperparameterRangeInteger(
        'b', lower_bound=2, upper_bound=3, scaling=LinearScaling())
    hp_c = HyperparameterRangeCategorical('c', choices=('1', '2', '3'))
    hp_d = HyperparameterRangeInteger(
        'd', lower_bound=2, upper_bound=3, scaling=LinearScaling())
    hp_e = HyperparameterRangeCategorical('e', choices=('1', '2'))

    for name_last_pos in ['a', 'c', 'd', 'e']:
        hp_ranges_cs = HyperparameterRanges_CS(
            config_space, name_last_pos=name_last_pos)
        if name_last_pos == 'a':
            lst = [hp_b, hp_c, hp_d, hp_e, hp_a]
        elif name_last_pos == 'c':
            lst = [hp_a, hp_b, hp_d, hp_e, hp_c]
        elif name_last_pos == 'd':
            lst = [hp_a, hp_b, hp_c, hp_e, hp_d]
        else:
            lst = [hp_a, hp_b, hp_c, hp_d, hp_e]
        hp_ranges = HyperparameterRanges_Impl(*lst)
        names = [hp.name for hp in hp_ranges.hp_ranges]
        config_cs = hp_ranges_cs.random_candidate(random_state)
        _config = config_cs.get_dictionary()
        config = (_config[name] for name in names)
        ndarr_cs = hp_ranges_cs.to_ndarray(config_cs)
        ndarr = hp_ranges.to_ndarray(config)
        assert_allclose(ndarr_cs, ndarr, rtol=1e-4)
