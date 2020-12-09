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

    This class supports the full range of parameters of the standard
    HyperbandScheduler. The 'objectives' replaces the 'reward_attr' parameter.
    Here we only explain the multi-objective specific parameters. For the full 
    list of available options refer too HyperbandScheduler.

    Parameters
    ----------
    objectives : dict
        Dictionary with the names of objectives of interest. The corresponding
        values are allowed to be either "MAX" or "MIN" and indicate if an 
        objective is to be maximized or minimized. The objective data is
        obtained from the reporter.
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
        kwargs["reward_attr"] = "_SCALARIZATION"
        super().__init__(train_fn, **filter_by_key(kwargs, _ARGUMENT_KEYS))


    def add_job(self, task, **kwargs):
        """Adding a training task to the scheduler.

        Args:
            task (:class:`autogluon.scheduler.Task`): a new training task

        Relevant entries in kwargs:
        - bracket: HB bracket to be used. Has been sampled in _promote_config
        - new_config: If True, task starts new config eval, otherwise it promotes
          a config (only if type == 'promotion')
        - elapsed_time: Time stamp
        Only if new_config == False:
        - config_key: Internal key for config
        - resume_from: config promoted from this milestone
        - milestone: config promoted to this milestone (next from resume_from)
        - scalarization_options: Specifies scalarazitation method to be used
        """
        cls = HyperbandScheduler
        if not self._delay_get_config:
            # Wait for resource to become available here, as this has not happened
            # in schedule_next before
            cls.resource_manager._request(task.resources)
        # reporter and terminator
        weights = [_uniform(len(self._objectives)) for _ in range(self._scalarization_options["num_weights"])]
        weights = [w * self._sign_vector for w in weights]
        reporter = MODistStatusReporter(self._objectives, weights, self._scalarization_options, remote=task.resources.node)
        task.args['reporter'] = reporter

        # TODO: Factor this part out by modifiying the main class
        # --------------------------------------------------------------------------------
        # Register task
        task_key = str(task.task_id)
        with self._hyperband_lock:
            assert task_key not in self._running_tasks, \
                "Task {} is already registered as running".format(task_key)
            self._running_tasks[task_key] = {
                'config': task.args['config'],
                'time_stamp': kwargs['elapsed_time'],
                'bracket': kwargs['bracket'],
                'reported_result': None,
                'keep_case': False}
            first_milestone = self.terminator.on_task_add(task, **kwargs)[-1]
        # Register pending evaluation(s) with searcher
        debug_log = self.searcher.debug_log
        if kwargs.get('new_config', True):
            # Task starts a new config
            next_milestone = first_milestone
            logger.debug("Adding new task (first milestone = {}):\n{}".format(
                next_milestone, task))
            if debug_log is not None:
                # Debug log output
                config_id = debug_log.config_id(task.args['config'])
                msg = "config_id {} starts (first milestone = {})".format(
                    config_id, next_milestone)
                logger.info(msg)
        else:
            # Promotion of config
            # This is a signal towards train_fn, which can be used for pause
            # & resume (given that train_fn checkpoints model state): Access
            # in train_fn as args.scheduler.resume_from
            if 'scheduler' not in task.args['args']:
                task.args['args']['scheduler'] = dict()
            task.args['args']['scheduler']['resume_from'] = kwargs['resume_from']
            next_milestone = kwargs['milestone']
            logger.debug("Promotion task (next milestone = {}):\n{}".format(
                next_milestone, task))
            if debug_log is not None:
                # Debug log output
                config_id = debug_log.config_id(task.args['config'])
                msg = "config_id {} promoted from {} (next milestone = {})".format(
                    config_id, kwargs['resume_from'], next_milestone)
                logger.info(msg)
        self.searcher.register_pending(
            task.args['config'], milestone=next_milestone)
        if self.maxt_pending and next_milestone != self.max_t:
            # Also introduce pending evaluation for resource max_t
            self.searcher.register_pending(
                task.args['config'], milestone=self.max_t)

        # main process
        job = cls._start_distributed_job(task, cls.resource_manager)
        # reporter thread
        rp = threading.Thread(
            target=self._run_reporter,
            args=(task, job, reporter),
            daemon=False)
        rp.start()
        task_dict = self._dict_from_task(task)
        task_dict.update({'Task': task, 'Job': job, 'ReporterThread': rp})
        # Checkpoint thread. This is also used for training_history
        # callback
        if self._checkpoint is not None or \
                self.training_history_callback is not None:
            self._add_checkpointing_to_job(job)
        with self.LOCK:
            self.scheduled_tasks.append(task_dict)

    def get_pareto_frontier(self):
        # This function is used to extract the search results from the scheduler results
        pass

def _prepare_sign_vector(objectives):
    """Generates a numpy vector which can be used to flip the signs of the objectives values
    which are intended to be minimized.

    Parameters
    ----------
    objectives: dict
        The dictionary keys name the objectives of interest. The associated values can be either
        "MIN" or "MAX" and indicate if an objective is to be minimized or maximized.

    Returns
    ----------
    sign_vector: np.array
        A numpy array containing 1 for objectives to be maximized and -1 for objectives to be
        minimized.
    """
    converter = {
        "MIN": -1.0,
        "MAX": 1.0
    }
    try:
        sign_vector = np.array([converter[objectives[k]] for k in objectives])
    except KeyError:
        raise ValueError("Error, in conversion of objective dict. Allowed values are 'MIN' and 'MAX'")
    return sign_vector


def _uniform(dim):
    """Samples a point uniformly at random from the unit simplex using the Kraemer Algorithm
    The algorithm is described here: https://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
    
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
    assert sum(sample) - 1 < 1e-6, "Error in weight sampling."
    return np.array(sample)
