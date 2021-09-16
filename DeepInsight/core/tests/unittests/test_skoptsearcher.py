import logging
import random
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from autogluon.core.searcher import RandomSearcher
from autogluon.core.searcher import SKoptSearcher

# Suppress known UserWarnings:
import warnings
warnings.filterwarnings("ignore", message=".*objective has been evaluated at this point before.*")
warnings.filterwarnings("ignore", message=".*skopt failed to produce new config, using random search instead.*")

logger = logging.getLogger(__name__)


def test_skoptsearcher():
    logger.debug('Start testing SKoptSearcher')
    random.seed(1)
    reward_attribute = 'accuracy'
    # Create configuration space:
    cs = CS.ConfigurationSpace()
    a = CSH.UniformFloatHyperparameter('a', lower=1e-4, upper=1e-1, log=True) # log-scale float
    b = CSH.UniformFloatHyperparameter('b', lower=-2, upper=0) # float with uniform prior
    c = CSH.UniformIntegerHyperparameter('c', lower=0, upper=1000) # integer
    d = CSH.CategoricalHyperparameter('d', choices=['good','neutral','bad']) # categorical
    cs.add_hyperparameters([a,b,c,d])
    # Determine reward of optimal config:
    optimal_config = cs.sample_configuration()
    optimal_config['a'] = 1e-1
    optimal_config['b'] = 0
    optimal_config['c'] = 1000
    optimal_config['d'] = 'good' 
    optimal_reward = toy_reward(optimal_config) # should ~= 7025.58
    # Compare skopt searchers VS random sampling searcher:
    num_configs_totry = 15
    skopt_searcher = SKoptSearcher(
        cs, reward_attribute=reward_attribute)
    skopt_config_list = [None]*num_configs_totry
    skopt_reward_list = [0.0]*num_configs_totry # stores rewards scaled between 0-1
    # Also try skopt searcher which uses various kwargs (random forest surrgoate model, expected improvement acquisition):
    skrf_searcher = SKoptSearcher(
        cs, reward_attribute=reward_attribute, base_estimator='RF',
        acq_func='EI')
    skrf_config_list = [None]*num_configs_totry 
    skrf_reward_list = [0.0]*num_configs_totry # stores rewards scaled between 0-1
    # Benchmark against random searcher:
    rs_searcher = RandomSearcher(cs, reward_attribute=reward_attribute)
    random_config_list = [None]*num_configs_totry
    random_reward_list = [0.0]*num_configs_totry
    # Run search:
    reported_result = {reward_attribute: 0.0}
    for i in range(num_configs_totry):
        skopt_config = skopt_searcher.get_config()
        skopt_reward = toy_reward(skopt_config) / optimal_reward
        reported_result[reward_attribute] = skopt_reward
        skopt_searcher.update(skopt_config, **reported_result)
        skopt_config_list[i] = skopt_config
        skopt_reward_list[i] = skopt_reward
        skrf_config = skrf_searcher.get_config()
        skrf_reward = toy_reward(skrf_config) / optimal_reward
        reported_result[reward_attribute] = skrf_reward
        skrf_searcher.update(skrf_config, **reported_result)
        skrf_config_list[i] = skrf_config
        skrf_reward_list[i] = skrf_reward
        rs_config = rs_searcher.get_config()
        rs_reward = toy_reward(rs_config) / optimal_reward
        reported_result[reward_attribute] = rs_reward
        rs_searcher.update(rs_config, **reported_result)
        random_config_list[i] = rs_config
        random_reward_list[i] = rs_reward
        # print("Round %d: skopt best reward=%f" % (i,max(skopt_reward_list)))
    # Summarize results:
    logger.debug("best reward from SKopt: %f,  best reward from SKopt w/ RF: %f,  best reward from Random search: %f" % 
          (max(skopt_reward_list), max(skrf_reward_list), max(random_reward_list)))
    # Ensure skopt outperformed random search:
    assert (max(skopt_reward_list) + 0.05 >= max(random_reward_list)), "SKopt did significantly worse than Random Search"
    # Ensure skopt found reasonably good config within num_configs_totry:
    assert (max(skopt_reward_list) >= 0.6), "SKopt performed poorly"
    logger.debug('Test Finished.')


def toy_reward(config):
    """ The reward function to maximize (ie. returns performance from a fake training trial).
    
        Args:
            config: dict() object defined in unit-test, not ConfigSpace object.
    """
    reward = 10*config['b'] + config['c']
    reward *= 30**config['a']
    if config['d'] == 'good':
        reward *= 5
    elif config['d'] == 'neutral':
        reward *= 2
    return reward

