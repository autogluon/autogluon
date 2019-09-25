import os
import json
import pickle
import logging
import threading
import multiprocessing as mp
from collections import OrderedDict

import mxnet as mx

from ..resource import DistributedResource
from ..basic import save, load
from ..utils import mkdir, try_import_mxboard
from ..basic import Task, autogluon_method
from ..searcher import RLSearcher
from .dist_scheduler import DistributedTaskScheduler
from .dist_fifo import DistributedFIFOScheduler
from .dist_reporter import DistStatusReporter

__all__ = ['DistributedRLScheduler']

logger = logging.getLogger(__name__)

class DistributedRLScheduler(DistributedFIFOScheduler):
    def __init__(self, train_fn, resource, checkpoint='./exp/checkerpoint.ag',
                 resume=False, num_trials=None, time_attr='epoch', reward_attr='accuracy',
                 visualizer='none', controller_lr=3.5e-4, ema_baseline_decay=0.95,
                 controller_resource={'num_cpus': 2, 'num_gpus': 0},
                 controller_batch_size=1,
                 dist_ip_addrs=[], **kwargs):
        assert isinstance(train_fn, autogluon_method), 'Please use autogluon.autogluon_register_args ' + \
            'to decorate your training script.'
        self.ema_baseline_decay = ema_baseline_decay
        # create RL searcher/controller
        searcher = RLSearcher(train_fn.get_kwspaces())
        super(DistributedRLScheduler,self).__init__(
                train_fn, train_fn.args, resource, searcher,
                checkpoint=checkpoint, resume=resume, num_trials=num_trials,
                time_attr=time_attr, reward_attr=reward_attr,
                visualizer=visualizer, dist_ip_addrs=dist_ip_addrs, **kwargs)
        # reserve controller computation resource on master node
        master_node = self.REMOTE_MANAGER.get_master_node()
        self.controller_resource = DistributedResource(**controller_resource)
        assert self.RESOURCE_MANAGER.reserve_resource(
                master_node, self.controller_resource), 'Not Enough Resource on Master Node' + \
                    ' for Training Controller'
        self.controller_ctx = [mx.gpu(i) for i in self.controller_resource.gpu_ids] if \
                controller_resource['num_gpus'] > 0 else [mx.cpu()]
        # controller setup
        self.controller = searcher.controller
        #self.controller.reset_ctx(self.controller_ctx)
        self.controller.initialize(ctx=self.controller_ctx)
        self.controller_optimizer = mx.gluon.Trainer(
                self.controller.collect_params(), 'adam',
                optimizer_params={'learning_rate': controller_lr*controller_batch_size})
        self.controller_batch_size = controller_batch_size

    def run(self, num_trials=None):
        """Run multiple number of trials
        """
        self.num_trials = num_trials if num_trials else self.num_trials
        logger.info('Starting Experiments')
        logger.info('Num of Finished Tasks is {}'.format(self.num_finished_tasks))
        logger.info('Num of Pending Tasks is {}'.format(self.num_trials - self.num_finished_tasks))

        baseline = None
        for i in range(self.num_trials // self.controller_batch_size + 1):
            with mx.autograd.record():
                # sample controller_batch_size number of configurations
                batch_size = self.num_trials % self.num_trials \
                    if i == self.num_trials // self.controller_batch_size \
                    else self.controller_batch_size
                if batch_size == 0: break
                configs, log_probs, entropies = self.controller.sample(
                    batch_size, with_details=True)

                # schedule the training tasks and gather the reward
                rewards = self.schedule_tasks(configs)

                # substract baseline
                if baseline is None:
                    baseline = rewards[0]

                avg_rewards = mx.nd.array([reward - baseline for reward in rewards],
                                          ctx=self.controller.context)

                # EMA baseline
                decay = self.ema_baseline_decay
                for reward in rewards:
                    baseline = decay * baseline + (1 - decay) * reward

                # negative policy gradient
                loss = -log_probs * avg_rewards.reshape(-1, 1)
                loss = loss.sum()  # or loss.mean()

            # update
            loss.backward()
            self.controller_optimizer.step(batch_size)
            logger.debug('controller loss: {}'.format(loss.asscalar()))

    def schedule_tasks(self, configs):
        rewards = []
        results = {}
        def _run_reporter(task, task_thread, reporter):
            last_result = None
            config = task.args['config']
            while task_thread.is_alive():
                reported_result = reporter.fetch()
                #print('reported_result', reported_result)
                if 'done' in reported_result and reported_result['done'] is True:
                    reporter.move_on()
                    task_thread.join()
                    break
                self.add_training_result(task.task_id, reported_result)
                reporter.move_on()
                last_result = reported_result
            if last_result is not None:
                self.searcher.update(config, last_result[self._reward_attr])
            results[pickle.dumps(config)] = last_result[self._reward_attr]

        # launch the tasks
        tasks = []
        task_threads = []
        reporter_threads = []
        for config in configs:
            logger.debug('scheduling config: {}'.format(config))
            # create task
            task = Task(self.train_fn, {'args': self.args, 'config': config},
                        DistributedResource(**self.resource))
            tasks.append(task)
            reporter = DistStatusReporter()
            task_thread = self.add_task(task, reporter)
            # run reporter
            reporter_thread = threading.Thread(target=_run_reporter, args=(task, task_thread, reporter))
            reporter_thread.start()
            task_threads.append(task_thread)
            reporter_threads.append(reporter_thread)

        for p1, p2 in zip(task_threads, reporter_threads):
            p1.join()
            p2.join()
        with self.LOCK:
            for task in tasks:
                self.finished_tasks.append({'TASK_ID': task.task_id,
                                           'Config': task.args['config']})
        if self._checkpoint is not None:
            logger.debug('Saving Checkerpoint')
            self.save()

        for config in configs:
            rewards.append(results[pickle.dumps(config)])

        return rewards

    def add_task(self, task, reporter):
        """Adding a training task to the scheduler.

        Args:
            task (:class:`autogluon.scheduler.Task`): a new trianing task
        """
        cls = DistributedFIFOScheduler
        cls.RESOURCE_MANAGER._request(task.resources)
        task.args['reporter'] = reporter
        # main process
        task_thread = threading.Thread(target=cls._start_distributed_task, args=(
                              task, cls.RESOURCE_MANAGER, self.env_sem))
        task_thread.start()
        return task_thread

    def join_tasks(self):
        pass

    def state_dict(self, destination=None):
        destination = super(DistributedFIFOScheduler, self).state_dict(destination)
        # TODO
        destination['searcher'] = pickle.dumps(self.searcher)
        destination['training_history'] = json.dumps(self.training_history)
        if self.visualizer == 'mxboard' or self.visualizer == 'tensorboard':
            destination['visualizer'] = json.dumps(self.mxboard._scalar_dict)
        return destination

    def load_state_dict(self, state_dict):
        super(DistributedFIFOScheduler, self).load_state_dict(state_dict)
        # TODO
        self.searcher = pickle.loads(state_dict['searcher'])
        self.training_history = json.loads(state_dict['training_history'])
        if self.visualizer == 'mxboard' or self.visualizer == 'tensorboard':
            self.mxboard._scalar_dict = json.loads(state_dict['visualizer'])
        logger.debug('Loading Searcher State {}'.format(self.searcher))
