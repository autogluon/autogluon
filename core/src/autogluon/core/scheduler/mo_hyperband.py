import logging
import threading
import numpy as np
import multiprocessing as mp

from .hyperband import HyperbandScheduler
from .reporter import MODistStatusReporter
from ..utils.default_arguments import filter_by_key
from ..utils.mo_hbo_utils import prepare_sign_vector, uniform_from_unit_simplex
from ..utils.mo_hbo_utils import retrieve_pareto_front


__all__ = ['MOHyperbandScheduler']

logger = logging.getLogger(__name__)

_ARGUMENT_KEYS = {
    'objectives', 'scalarization_options'
}


class MOHyperbandScheduler(HyperbandScheduler):
    r"""Implements different variants of asynchronous multi-objective Hyperband
    by extending the standard HyperbandScheduler using random scalarizations. 
    The implemented method is described in detail in:

        Schmucker, Donini, Perrone, Zafar, Archambeau (2020)
        Multi-Objective Multi-Fidelity Hyperparameter Optimization with 
        Application to Fairness
        https://meta-learn.github.io/2020/papers/24_paper.pdf

    This class supports the full range of parameters of the standard
    HyperbandScheduler. The 'objectives' replaces the 'reward_attr' parameter.
    Here we only explain the multi-objective specific parameters. For the full 
    list of available options refer to HyperbandScheduler.

    Parameters
    ----------
    objectives : dict
        Dictionary with the names of objectives of interest. The corresponding
        values are allowed to be either "MAX" or "MIN" and indicate if an 
        objective is to be maximized or minimized.
    scalarization_options : dict
        Contains arguments for the underlying scalarization algorithm. 
        Available algorithms are "random_weights" and "parego".

    Examples
    --------
    examples/mo_hpo/mo_hyperband.ipynb
    """

    def __init__(self, train_fn, **kwargs):
        self._scalarization_options = kwargs["scalarization_options"]
        self._objectives = kwargs["objectives"]
        self._sign_vector = prepare_sign_vector(self._objectives)
        assert "_SCALARIZATION" not in kwargs["objectives"], "_SCALARIZATION" \
            "is a protected value. Please refrain from using it."
        kwargs["reward_attr"] = "_SCALARIZATION"
        super().__init__(train_fn, **filter_by_key(kwargs, _ARGUMENT_KEYS))

    def add_job(self, task, **kwargs):
        """Adding a training task to the scheduler.

        Args:
            task (:class:`autogluon.scheduler.Task`): a new training task
        """
        n_weights = self._scalarization_options.get('num_weights', 100)
        weights = [uniform_from_unit_simplex(len(self._objectives)) for _ in range(n_weights)]
        weights = [w * self._sign_vector for w in weights]
        reporter = MODistStatusReporter(self._objectives, weights, 
                self._scalarization_options, remote=task.resources.node)
        kwargs["reporter"] = reporter
        super().add_job(task, **kwargs)

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
        pareto_front = retrieve_pareto_front(self.training_history, self._objectives)
        return pareto_front
