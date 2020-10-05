import pickle
import logging

from ..utils import warning_filter
with warning_filter():
    from skopt import Optimizer
    from skopt.space import Integer, Real, Categorical

from .searcher import BaseSearcher

__all__ = ['SKoptSearcher']

logger = logging.getLogger(__name__)


class SKoptSearcher(BaseSearcher):
    """SKopt Searcher that uses Bayesian optimization to suggest new hyperparameter configurations. 
        Requires that 'scikit-optimize' package is installed.
    
    Parameters
    ----------
    configspace: ConfigSpace.ConfigurationSpace
        The configuration space to sample from. It contains the full
        specification of the Hyperparameters with their priors
    kwargs: Optional arguments passed to skopt.optimizer.Optimizer class. 
        Please see documentation at this link: `skopt.optimizer.Optimizer <http://scikit-optimize.github.io/optimizer/index.html#skopt.optimizer.Optimizer>`_
        These kwargs be used to specify which surrogate model Bayesian optimization should rely on,
        which acquisition function to use, how to optimize the acquisition function, etc.
        The skopt library provides comprehensive Bayesian optimization functionality,
        where popular non-default kwargs options here might include: 
        
        - base_estimator = 'GP' or 'RF' or 'ET' or 'GBRT' (to specify different surrogate models like Gaussian Processes, Random Forests, etc)
        
        - acq_func = 'LCB' or 'EI' or 'PI' or 'gp_hedge' (to specify different acquisition functions like Lower Confidence Bound, Expected Improvement, etc)
        
        For example, we can tell our Searcher to perform Bayesian optimization with a Random Forest surrogate model
        and use the Expected Improvement acquisition function by invoking: 
        `SKoptSearcher(cs, base_estimator='RF', acq_func='EI')`
    
    Examples
    --------
    By default, the searcher is created along with the scheduler. For example:

    >>> import autogluon.core as ag
    >>> @ag.args(
    ...     lr=ag.space.Real(1e-3, 1e-2, log=True))
    >>> def train_fn(args, reporter):
    ...     reporter(accuracy = args.lr ** 2)
    >>> search_options = {'base_estimator': 'RF', 'acq_func': 'EI'}
    >>> scheduler = ag.scheduler.FIFOScheduler(
    ...     train_fn, searcher='skopt', search_options=search_options,
    ...     num_trials=10, reward_attr='accuracy')

    This would result in a SKoptSearcher with cs = train_fn.cs. You can also
    create a SKoptSearcher by hand:

    >>> import autogluon.core as ag
    >>> @ag.args(
    ...     lr=ag.space.Real(1e-3, 1e-2, log=True),
    ...     wd=ag.space.Real(1e-3, 1e-2))
    >>> def train_fn(args, reporter):
    ...     pass
    >>> searcher = ag.searcher.SKoptSearcher(train_fn.cs)
    >>> searcher.get_config()
    {'lr': 0.0031622777, 'wd': 0.0055}
    >>> searcher = SKoptSearcher(
    ...     train_fn.cs, reward_attribute='accuracy', base_estimator='RF',
    ...     acq_func='EI')
    >>> next_config = searcher.get_config()
    >>> searcher.update(next_config, accuracy=10.0)  # made-up value
    
    .. note::
        
        - get_config() cannot ensure valid configurations for conditional spaces since skopt 
        does not contain this functionality as it is not integrated with ConfigSpace. 
        If invalid config is produced, `SKoptSearcher.get_config()` will catch these Exceptions and revert to `random_config()` instead.
        
        - get_config(max_tries) uses skopt's batch BayesOpt functionality to query at most 
        max_tries number of configs to try out.
        If all of these have configs have already been scheduled to try (might happen in asynchronous setting), 
        then get_config simply reverts to random search via random_config().
    """
    errors_tohandle = (ValueError, TypeError, RuntimeError)

    def __init__(self, configspace, **kwargs):
        super().__init__(
            configspace, reward_attribute=kwargs.get('reward_attribute'))
        self.hp_ordering = configspace.get_hyperparameter_names() # fix order of hyperparams in configspace.
        skopt_hpspace = []
        for hp in self.hp_ordering:
            hp_obj = configspace.get_hyperparameter(hp)
            hp_type = str(type(hp_obj)).lower() # type of hyperparam
            if 'integer' in hp_type:
                hp_dimension = Integer(low=int(hp_obj.lower), high=int(hp_obj.upper),name=hp)
            elif 'float' in hp_type:
                if hp_obj.log: # log10-scale hyperparmeter
                    hp_dimension = Real(low=float(hp_obj.lower), high=float(hp_obj.upper), prior='log-uniform', name=hp)
                else:
                    hp_dimension = Real(low=float(hp_obj.lower), high=float(hp_obj.upper), name=hp)
            elif 'categorical' in hp_type:
                hp_dimension = Categorical(hp_obj.choices, name=hp)
            elif 'ordinal' in hp_type:
                hp_dimension = Categorical(hp_obj.sequence, name=hp)
            else:
                raise ValueError("unknown hyperparameter type: %s" % hp)
            skopt_hpspace.append(hp_dimension)
        skopt_keys = {
            'base_estimator', 'n_random_starts', 'n_initial_points',
            'acq_func', 'acq_optimizer', 'random_state',  'model_queue_size',
            'acq_func_kwargs', 'acq_optimizer_kwargs'}
        skopt_kwargs = self._filter_skopt_kwargs(kwargs, skopt_keys)
        self.bayes_optimizer = Optimizer(
            dimensions=skopt_hpspace, **skopt_kwargs)

    @staticmethod
    def _filter_skopt_kwargs(kwargs, keys):
        return {k: v for k, v in kwargs.items() if k in keys}

    def configure_scheduler(self, scheduler):
        from ..scheduler import FIFOScheduler

        assert isinstance(scheduler, FIFOScheduler), \
            "This searcher requires FIFOScheduler scheduler"
        super().configure_scheduler(scheduler)

    def get_config(self, **kwargs):
        """Function to sample a new configuration
        This function is called to query a new configuration that has not yet been tried.
        Asks for one point at a time from skopt, up to max_tries. 
        If an invalid hyperparameter configuration is proposed by skopt, then reverts to random search
        (since skopt configurations cannot handle conditional spaces like ConfigSpace can).
        TODO: may loop indefinitely due to no termination condition (like RandomSearcher.get_config() ) 
        
        Parameters
        ----------
        max_tries: int, default = 1e2
            The maximum number of tries to ask for a unique config from skopt before reverting to random search.
        """
        max_tries = kwargs.get('max_tries', 1e2)
        if len(self._results) == 0: # no hyperparams have been tried yet, first try default config
            return self.default_config()
        try:
            new_points = self.bayes_optimizer.ask(n_points=1) # initially ask for one new config
            new_config_cs = self.skopt2config(new_points[0]) # hyperparameter-config to evaluate
            try:
                new_config_cs.is_valid_configuration()
                new_config = new_config_cs.get_dictionary()
                if pickle.dumps(new_config) not in self._results.keys(): # have not encountered this config
                    self._results[pickle.dumps(new_config)] = self._reward_while_pending()
                    return new_config
            except self.errors_tohandle:
                pass
            new_points = self.bayes_optimizer.ask(n_points=max_tries) # ask skopt for many configs since first one was not new
            i = 1 # which new point to return as new_config, we already tried the first point above
            while i < max_tries:
                new_config_cs = self.skopt2config(new_points[i]) # hyperparameter-config to evaluate
                try:
                    new_config_cs.is_valid_configuration()
                    new_config = new_config_cs.get_dictionary()
                    if (pickle.dumps(new_config) not in self._results.keys()): # have not encountered this config
                        self._results[pickle.dumps(new_config)] = self._reward_while_pending()
                        return new_config
                except self.errors_tohandle:
                    pass
                i += 1
        except self.errors_tohandle:
            pass
        logger.info("used random search instead of skopt to produce new hyperparameter configuration in this trial")
        return self.random_config()
    
    def default_config(self):
        """ Function to return the default configuration that should be tried first.
        
        Returns
        -------
        returns: config
        """
        new_config_cs = self.configspace.get_default_configuration()
        new_config = new_config_cs.get_dictionary()
        self._results[pickle.dumps(new_config)] = self._reward_while_pending()
        return new_config
        
    def random_config(self):
        """Function to randomly sample a new configuration (which is ensured to be valid in the case of conditional hyperparameter spaces).
        """
        # TODO: may loop indefinitely due to no termination condition (like RandomSearcher.get_config() ) 
        new_config = self.configspace.sample_configuration().get_dictionary()
        while pickle.dumps(new_config) in self._results.keys():
            new_config = self.configspace.sample_configuration().get_dictionary()
        self._results[pickle.dumps(new_config)] = self._reward_while_pending()
        return new_config

    def update(self, config, **kwargs):
        """Update the searcher with the newest metric report.
        """
        super().update(config, **kwargs)
        reward = kwargs[self._reward_attribute]
        try:
            self.bayes_optimizer.tell(self.config2skopt(config),
                                      -reward)  # provide negative reward since skopt performs minimization
        except self.errors_tohandle:
            logger.info("surrogate model not updated this trial")

    def config2skopt(self, config):
        """ Converts autogluon config (dict object) to skopt format (list object).

        Returns
        -------
        Object of same type as: `skOpt.Optimizer.ask()`
        """
        point = []
        for hp in self.hp_ordering:
            point.append(config[hp])
        return point
    
    def skopt2config(self, point):
        """ Converts skopt point (list object) to autogluon config format (dict object. 
        
        Returns
        -------
        Object of same type as: `RandomSampling.configspace.sample_configuration().get_dictionary()`
        """
        config = self.configspace.sample_configuration()
        for i in range(len(point)):
            hp = self.hp_ordering[i]
            config[hp] = point[i]
        return config
