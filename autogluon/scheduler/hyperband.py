import pickle
import logging
import threading
import numpy as np
import multiprocessing as mp

from .fifo import FIFOScheduler
from .hyperband_stopping import HyperbandStopping_Manager
from .hyperband_promotion import HyperbandPromotion_Manager
from .reporter import DistStatusReporter, DistSemaphore
from ..utils import DeprecationHelper

__all__ = ['HyperbandScheduler', 'DistributedHyperbandScheduler']

logger = logging.getLogger(__name__)


class HyperbandScheduler(FIFOScheduler):
    """Implements different variants of asynchronous Hyperband

    See 'type' for the different variants. One implementation detail is when
    using multiple brackets, task allocation to bracket is done randomly with
    over a softmax probability.

    Args:
        train_fn (callable): A task launch function for training.
            Note: please add the `@autogluon_method` decorater to the original function.
        args (object): Default arguments for launching train_fn.
        resource (dict): Computation resources.
            For example, `{'num_cpus':2, 'num_gpus':1}`
        searcher (object): Autogluon searcher.
            For example, autogluon.searcher.RandomSearcher
        time_attr (str): A training result attr to use for comparing time.
            Note that you can pass in something non-temporal such as
            `training_epoch` as a measure of progress, the only requirement
            is that the attribute should increase monotonically.
        reward_attr (str): The training result objective value attribute.
            As with `time_attr`, this may refer to any objective value.
            Stopping procedures will use this attribute.
        max_t (float): max time units per task.
            Trials will be stopped after max_t time units (determined by
            time_attr) have passed.
        grace_period (float): Only stop tasks at least this old in time.
            Also: min_t. The units are the same as the attribute named by
            `time_attr`.
        reduction_factor (float): Used to set halving rate and amount.
            This is simply a unit-less scalar.
        brackets (int): Number of brackets. Each bracket has a different
            grace period, all share max_t and reduction_factor.
            If brackets == 1, we just run successive halving, for
            brackets > 1, we run Hyperband.
        type (str): Type of Hyperband scheduler:
            stopping: See HyperbandStopping_Manager. Tasks and config evals are
                tightly coupled. A task is stopped at a milestone if worse than
                most others, otherwise it continues. As implemented in Ray/Tune:
                https://ray.readthedocs.io/en/latest/tune-schedulers.html#asynchronous-hyperband
            promotion: See HyperbandPromotion_Manager. A config eval may be
                associated with multiple tasks over its lifetime. It is never
                terminated, but may be paused. Whenever a task becomes available,
                it may promote a config to the next milestone, if better than most
                others. If no config can be promoted, a new one is chosen. This
                variant may benefit from pause&resume, which is not directly
                supported here. As proposed in this paper (termed ASHA):
                https://arxiv.org/abs/1810.05934
        keep_size_ratios (bool): Implemented for type 'promotion' only. If True,
            promotions are done only if the (current estimate of the) size ratio
            between rung and next rung are 1 / reduction_factor or better. This
            avoids higher rungs to get more populated than they would be in
            synchronous Hyperband. A drawback is that promotions to higher rungs
            take longer.
        maxt_pending (bool): Relevant only if a model-based searcher is used.
            If True, register pending config at level max_t
            whenever a new evaluation is started. This has a direct effect on
            the acquisition function (for model-based variant), which operates
            at level max_t. On the other hand, it decreases the variance of the
            latent process there.
            NOTE: This could also be removed...
        searcher_data (str): Ways for searcher to maintain observed data:
            Determines what observations (config, result), fetched in
            _run_reporter, are collected in the searcher dataset (only relevant
            for searchers that maintain a dataset).
            rungs (default): Maintain results if time_attr value is equal to a
                rung level.
            all: Maintain all results reported to _run_reporter.
            rungs_and_last: Maintain results as for rungs (at rung levels), and
                also the result (for each config) with the largest time_attr
                value.

    Example:
        >>> @autogluon_method
        >>> def train_fn(args, reporter):
        >>>     for e in range(10):
        >>>         # forward, backward, optimizer step and evaluation metric
        >>>         # generate fake top1_accuracy
        >>>         top1_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
        >>>         reporter(epoch=e, accuracy=top1_accuracy)
        >>> import ConfigSpace as CS
        >>> import ConfigSpace.hyperparameters as CSH
        >>> cs = CS.ConfigurationSpace()
        >>> lr = CSH.UniformFloatHyperparameter('lr', lower=1e-4, upper=1e-1, log=True)
        >>> cs.add_hyperparameter(lr)
        >>> searcher = RandomSampling(cs)
        >>> myscheduler = HyperbandScheduler(train_fn, args,
        >>>                                  resource={'num_cpus': 2, 'num_gpus': 0}, 
        >>>                                  searcher=searcher, num_trials=20,
        >>>                                  reward_attr='accuracy',
        >>>                                  time_attr='epoch',
        >>>                                  grace_period=1)
    """
    def __init__(self, train_fn, args=None, resource=None,
                 searcher='random', search_options=None,
                 checkpoint='./exp/checkerpoint.ag',
                 resume=False, num_trials=None,
                 time_out=None, max_reward=1.0,
                 time_attr="epoch",
                 reward_attr="accuracy",
                 max_t=100, grace_period=10,
                 reduction_factor=4, brackets=1,
                 visualizer='none',
                 type='stopping',
                 dist_ip_addrs=None,
                 keep_size_ratios=False,
                 maxt_pending=False,
                 searcher_data='rungs'):
        super(HyperbandScheduler, self).__init__(
            train_fn=train_fn, args=args, resource=resource, searcher=searcher,
            search_options=search_options, checkpoint=checkpoint, resume=resume,
            num_trials=num_trials, time_out=time_out, max_reward=max_reward, time_attr=time_attr,
            reward_attr=reward_attr, visualizer=visualizer, dist_ip_addrs=dist_ip_addrs)
        self.max_t = max_t
        self.type = type
        self.maxt_pending = maxt_pending
        if type == 'stopping':
            self.terminator = HyperbandStopping_Manager(
                time_attr, reward_attr, max_t, grace_period, reduction_factor,
                brackets)
        elif type == 'promotion':
            self.terminator = HyperbandPromotion_Manager(
                time_attr, reward_attr, max_t, grace_period, reduction_factor,
                brackets, keep_size_ratios=keep_size_ratios)
        else:
            raise AssertionError(
                "type '{}' not supported, must be 'stopping' or 'promotion'".format(
                    type))
        sd_values = {'rungs', 'all', 'rungs_and_last'}
        assert searcher_data in sd_values, \
            "searcher_data = '{}', must be in {}".format(
                searcher_data, sd_values)
        self.searcher_data = searcher_data
        # Only maintained if searcher_data == 'rungs_and_last':
        # Maps task_id -> (config, reported_result, keep_case)
        # Here, reported_result is the last recent result for config for which
        # self.searcher.update has been called by task task_id (in
        # _run_reporter). This allows us to remove this case from the dataset
        # maintained by the searcher, once the next result for this config
        # comes in. If keep_case == True, the case is not removed though.
        self.last_recent_update = dict()

    def add_task(self, task, **kwargs):
        """Adding a training task to the scheduler.

        Args:
            task (:class:`autogluon.scheduler.Task`): a new trianing task

        Relevant entries in kwargs:
        - bracket: HB bracket to be used. Has been sampled in _promote_config
        - new_config: If True, task starts new config eval, otherwise it promotes
          a config (only if type == 'promotion')
        Only if new_config == False:
        - config_key: Internal key for config
        - resume_from: config promoted from this milestone
        - milestone: config promoted to this milestone (next from resume_from)
        """
        cls = HyperbandScheduler
        cls.RESOURCE_MANAGER._request(task.resources)
        # reporter and terminator
        reporter = DistStatusReporter()
        terminator_semaphore = DistSemaphore(0)
        task.args['reporter'] = reporter
        task.args['terminator_semaphore'] = terminator_semaphore

        # NOTE: If search_data != 'rungs', the pending candidates chosen here
        # may not correspond to the next recent data obtained in
        # _run_reporter (in particular, they all lie on rung levels). We ignore
        # this for sake of simplicity. Once the next result is recorded for
        # this task, everything will be OK again.
        # It is also not a problem that pending candidates registered here,
        # will be registered once more in the future, as this is permitted
        # by register_pending.
        milestones = self.terminator.on_task_add(task, **kwargs)
        if kwargs.get('new_config', True):
            first_milestone = milestones[-1]
            logger.debug("Adding new task (first milestone = {}):\n{}".format(
                first_milestone, task))
            self.searcher.register_pending(
                task.args['config'], first_milestone)
            if self.maxt_pending:
                # Also introduce pending evaluation for resource max_t
                final_milestone = milestones[0]
                if final_milestone != first_milestone:
                    self.searcher.register_pending(
                        task.args['config'], final_milestone)
        else:
            # Promotion of config
            # This is a signal towards train_fn, in case it supports
            # pause and resume:
            task.args['resume_from'] = kwargs['resume_from']
            next_milestone = kwargs['milestone']
            logger.debug("Promotion task (next milestone = {}):\n{}".format(
                next_milestone, task))
            self.searcher.register_pending(
                task.args['config'], next_milestone)

        # main process
        tp = threading.Thread(target=cls._start_distributed_task, args=(
                              task, cls.RESOURCE_MANAGER, self.env_sem))
        # reporter thread
        checkpoint_semaphore = mp.Semaphore(0) if self._checkpoint else None
        rp = threading.Thread(target=self._run_reporter,
                              args=(task, tp, reporter, self.searcher, self.terminator,
                                    checkpoint_semaphore, terminator_semaphore), daemon=False)
        tp.start()
        rp.start()
        task_dict = self._dict_from_task(task)
        task_dict.update({'Task': task, 'Process': tp, 'ReporterThread': rp})
        # checkpoint thread
        if self._checkpoint is not None:
            sp = threading.Thread(target=self._run_checkpoint, args=(checkpoint_semaphore,),
                                  daemon=False)
            sp.start()
            task_dict['CheckpointThead'] = sp

        with self.LOCK:
            self.scheduled_tasks.append(task_dict)

    def _update_searcher(self, searcher, task, result, keep_case):
        config = task.args['config']
        searcher.update(
            config=config, reward=result[self._reward_attr], **result)
        if self.searcher_data == 'rungs_and_last':
            self.last_recent_update[task.task_id] = (config, result, keep_case)

    def _run_reporter(self, task, task_process, reporter, searcher, terminator,
                      checkpoint_semaphore, terminator_semaphore):
        last_result = None
        last_updated = None
        while task_process.is_alive():
            reported_result = reporter.fetch()
            if reported_result.get('done', False):
                reporter.move_on()
                terminator_semaphore.release()
                terminator.on_task_complete(task, last_result)
                task_process.join()
                if checkpoint_semaphore is not None:
                    checkpoint_semaphore.release()
                break
            # Call before add_training_results, since we may be able to report
            # extra information from the bracket
            task_continues, update_searcher, next_milestone, bracket_id, rung_counts = \
                terminator.on_task_report(task, reported_result)
            reported_result['bracket'] = bracket_id
            if rung_counts is not None:
                for k, v in rung_counts.items():
                    key = 'count_at_{}'.format(k)
                    reported_result[key] = v
            self.add_training_result(
                task.task_id, reported_result, config=task.args['config'])
            if self.searcher_data == 'rungs':
                # Only results on rung levels are reported to the searcher
                if update_searcher and task_continues:
                    # Update searcher with intermediate result
                    # Note: If task_continues is False here, we also call
                    # searcher.update, but outside the loop.
                    self._update_searcher(
                        searcher, task, reported_result, True)
                    last_updated = reported_result
                    if next_milestone is not None:
                        searcher.register_pending(
                            task.args['config'], next_milestone)
            else:
                # All results are reported to the searcher
                if self.searcher_data == 'rungs_and_last' and \
                        task.task_id in self.last_recent_update:
                    # Remove last recently added result for this task, unless
                    # it fell on a rung level (i.e., keep_case)
                    rem_config, rem_result, keep_case = \
                        self.last_recent_update[task.task_id]
                    if not keep_case:
                        searcher.remove_case(
                            config=rem_config,
                            reward=rem_result[self._reward_attr],
                            **rem_result)
                # If searcher_data == 'rungs_and_last', the result is kept in
                # the dataset iff update_searcher == True
                self._update_searcher(
                    searcher, task, reported_result, update_searcher)
                last_updated = reported_result
                # Since all results are reported, the next report for this task
                # will be for resource + 1.
                # NOTE: This assumes that results are reported for all successive
                # resource levels (int). If any resource level is skipped,
                # there may be left-over pending candidates, who will be
                # removed once the task finishes.
                if task_continues:
                    searcher.register_pending(
                        task.args['config'],
                        int(reported_result[self._time_attr]) + 1)

            last_result = reported_result
            if task_continues:
                reporter.move_on()
            else:
                # Note: The 'terminated' signal is sent even in the promotion
                # variant. It means that the *task* terminates, while the evaluation
                # of the config is just paused
                last_result['terminated'] = True
                if self.type == 'stopping' or \
                        reported_result[self._time_attr] >= self.max_t:
                    act_str = 'terminating'
                else:
                    act_str = 'pausing'
                logger.debug(
                    'Stopping task ({} evaluation, resource = {}):\n{}'.format(
                        act_str, reported_result[self._time_attr], task))
                terminator_semaphore.release()
                terminator.on_task_remove(task)
                task_process.join()
                if checkpoint_semaphore is not None:
                    checkpoint_semaphore.release()
                break
        # Pass all of last_result to searcher (unless this has already been
        # done)
        if last_result is not last_updated:
            self._update_searcher(searcher, task, last_result, True)

    def state_dict(self, destination=None):
        destination = super(HyperbandScheduler, self).state_dict(destination)
        destination['terminator'] = pickle.dumps(self.terminator)
        return destination

    def load_state_dict(self, state_dict):
        super(HyperbandScheduler, self).load_state_dict(state_dict)
        self.terminator = pickle.loads(state_dict['terminator'])
        logger.info('Loading Terminator State {}'.format(self.terminator))

    def _promote_config(self):
        config, extra_kwargs = self.terminator.on_task_schedule()
        return config, extra_kwargs

    def map_resource_to_index(self):
        def fun(resource):
            assert 0.0 <= resource <= 1.0, \
                "resource must be in [0, 1]"
            return self.terminator._resource_to_index(resource)

        return fun

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' +  \
            'terminator: ' + str(self.terminator)
        return reprstr


DistributedHyperbandScheduler = DeprecationHelper(HyperbandScheduler, 'DistributedHyperbandScheduler')
