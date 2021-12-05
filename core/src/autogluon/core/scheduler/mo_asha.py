import logging
import copy
import numpy as np

from .hyperband import HyperbandScheduler, HyperbandBracketManager
from .mo_asha_promotion import MOPromotionRungSystem, EPS_NET, NSGA_II
from ..utils.default_arguments import Integer, Boolean, Categorical, \
    filter_by_key
from ..utils.mo_hbo_utils import prepare_sign_vector
from ..utils.mo_hbo_utils import retrieve_pareto_front

__all__ = ['MOASHAScheduler']

logger = logging.getLogger(__name__)

_ARGUMENT_KEYS = {
    'objectives',
}


class MOASHAScheduler(HyperbandScheduler):
    r"""Implements different variants of asynchronous multi-objective Hyperband
    by extending the standard HyperbandScheduler using a multi-objective
    version of ASHA. This version internally ranks candidates by first sorting
    them into different Pareto frontiers and then spaces them out evenly based
    on euclidean distance.

    Parameters
    ----------
    objectives : dict
        Dictionary with the names of objectives of interest. The corresponding
        values are allowed to be either "MAX" or "MIN" and indicate if an
        objective is to be maximized or minimized.

    Examples
    --------
    examples/mo_hpo/mo_asha.ipynb
    """

    _CONSTRAINTS = {
        'resume': Boolean(),
        'max_t': Integer(1, None),
        'grace_period': Integer(1, None),
        'reduction_factor': Integer(2, None),
        'brackets': Integer(1, None),
        'type': Categorical((EPS_NET, NSGA_II)),
        'searcher_data': Categorical(
            ('rungs', 'all', 'rungs_and_last')),
        'do_snapshots': Boolean(),
        'rung_system_per_bracket': Boolean(),
        'random_seed': Integer(0, None)}

    def __init__(self, train_fn, **kwargs):
        self._objectives = kwargs["objectives"]
        assert kwargs["type"] in self._CONSTRAINTS["type"].choices, \
            "Only eps_net and nsga_ii are supported"
        # NOTE: This is just a dummy objective
        kwargs["reward_attr"] = list(self._objectives.keys())[0]
        super().__init__(train_fn, **filter_by_key(kwargs, _ARGUMENT_KEYS))

    def _init_bracket_manager(self, scheduler_type, max_t, rung_levels,
                              brackets, rung_system_per_bracket, random_seed):
        # Initialize bracket manager.
        terminator = MOHyperbandBracketManager(
            scheduler_type, self._time_attr, self._reward_attr, max_t,
            rung_levels, brackets, rung_system_per_bracket,
            random_seed, self._objectives)
        return terminator

    def get_pareto_front(self):
        """Retrieves the pareto efficient points discovered during the search
        process. Raises an error if called before search was conducted.

        Returns
        ----------
        front: list
            A list containing the Pareto efficient points among all points
            that were found during the search process.
        """
        assert len(self.finished_tasks) > 0, "Can only extract front after \
            jobs have been completed."
        pareto_front = retrieve_pareto_front(self.training_history,
                                             self._objectives)
        return pareto_front


class MOHyperbandBracketManager(HyperbandBracketManager):
    """MO-Hyperband Bracket Manager

    Arguments follow the parent class

    Args:
        objectives : dict
            Indicates objectives of interest for the rung system
    """
    def __init__(
            self, scheduler_type, time_attr, reward_attr, max_t, rung_levels,
            brackets, rung_system_per_bracket, random_seed, objectives):
        # Sign vector ensures that we deal with a pure minimization problem
        self.sign_vector = -prepare_sign_vector(objectives)
        self.objectives = objectives
        super().__init__(scheduler_type, time_attr, reward_attr, max_t,
                         rung_levels, brackets, rung_system_per_bracket,
                         random_seed)

    @staticmethod
    def _init_rung_systems(scheduler_type, rung_levels, promote_quantiles,
                           max_t, num_systems):
        # Initialize MOPromotionRungSystem rung system for each level.
        rung_systems = [
            MOPromotionRungSystem(rung_levels[s:], promote_quantiles[s:], max_t, scheduler_type)
            for s in range(num_systems)
        ]
        return rung_systems

    def _prepare_objective_vector(self, result):
        values = np.array([result[k] for k in self.objectives])
        values = values * self.sign_vector
        return values

    def on_task_report(self, task, result):
        """Passes a modified result dictionary to the parent class which causes
        the bracket manager to pass a numpy array containing multiple objective
        values to the rung system.

        :param task: Only task.task_id is used
        :param result: Current reported results from task
        :return: See parent class
        """
        result_copy = copy.copy(result)
        result_copy["_REWARD_VECTOR"] = \
            self._prepare_objective_vector(result_copy)
        return super().on_task_report(task, result_copy)

    def _report_to_rung_sys(self, rung_sys, task, result, skip_rungs):
        # Pass result to rung system to decide if task may continue.
        rung_info = rung_sys.on_task_report(
                task, result[self._time_attr], result["_REWARD_VECTOR"],
                skip_rungs=skip_rungs)
        return rung_info
