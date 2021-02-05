import logging
import threading
import numpy as np
import multiprocessing as mp

from .hyperband import HyperbandScheduler
from .reporter import MODistStatusReporter
from ..utils.default_arguments import filter_by_key

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
        self._sign_vector = _prepare_sign_vector(self._objectives)
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
        weights = [_uniform(len(self._objectives)) for _ in range(n_weights)]
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
        vals = []
        refs = []

        for task_id, task_res in self.training_history.items():
            for step, res_dict in enumerate(task_res):
                vals.append([res_dict[k] for k in self._objectives])
                refs.append((task_id, step + 1))
        vals = np.array(vals)

        # We determine the Pareto front assuming pure minimization we adapt
        # the sign accordingly
        a_vals = vals * (-self._sign_vector)
        eff_mask = np.ones(vals.shape[0], dtype=bool)
        for i, c in enumerate(a_vals):
            if eff_mask[i]:
                eff_mask[eff_mask] = np.any(a_vals[eff_mask] <= c, axis=1)
        indices = [i for i, b in enumerate(eff_mask) if b]

        front = []
        for e in indices:
            r = {"task_id-ressource": refs[e]}
            for i, o in enumerate(self._objectives):
                r[o] = vals[e][i]
            front.append(r)
        return front


def _prepare_sign_vector(objectives):
    """Generates a numpy vector which can be used to flip the signs of the
    objectives values which are intended to be minimized.

    Parameters
    ----------
    objectives: dict
        The dictionary keys name the objectives of interest. The associated
        values can be either "MIN" or "MAX" and indicate if an objective is 
        to be minimized or maximized.

    Returns
    ----------
    sign_vector: np.array
        A numpy array containing 1 for objectives to be maximized and -1 for
        objectives to be minimized.
    """
    converter = {
        "MIN": -1.0,
        "MAX": 1.0
    }
    try:
        sign_vector = np.array([converter[objectives[k]] for k in objectives])
    except KeyError:
        raise ValueError("Error, in conversion of objective dict. Allowed \
            values are 'MIN' and 'MAX'")
    return sign_vector


def _uniform(dim):
    """Samples a point uniformly at random from the unit simplex using the
    Kraemer Algorithm. The algorithm is described here:
    https://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
    
    Parameters
    ----------
    dim: int
        Dimension of the unit simplex to sample from.
    
    Returns:
    sample: np.array
         A point sampled uniformly from the unit simplex.
    """
    uni = np.random.uniform(size=(dim))
    uni = np.sort(uni)
    sample =  np.diff(uni, prepend=0) / uni[-1]
    assert sum(sample) - 1 < 1e-6, "Error in weight sampling routine."
    return np.array(sample)
