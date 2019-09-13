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
                 controller_resource={'num_cpus': 2, 'num_gpus': 1},
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
                optimizer_params={'learning_rate': controller_lr})
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
            print('controller loss: {}'.format(loss.asscalar()))

    def schedule_tasks(self, configs):
        rewards = []
        # launch the tasks
        for config in configs:
            task = Task(self.train_fn, {'args': self.args, 'config': config},
                        DistributedResource(**self.resource))
            print('scheduling config: {}'.format(config))
            self.add_task(task)

        # join tasks and gather the result
        self.join_tasks()
        for config in configs:
            rewards.append(self.searcher.get_reward(config))

        return rewards

    def join_tasks(self):
        for i, task_dict in enumerate(self.scheduled_tasks):
            task_dict['Process'].join()
            task_dict['ReporterThread'].join()

        for i in range(len(self.scheduled_tasks)):
            task_dict = self.scheduled_tasks.pop()
            self.finished_tasks.append({'TASK_ID': task_dict['TASK_ID'],
                                       'Config': task_dict['Config']})

    def state_dict(self, destination=None):
        destination = super(DistributedFIFOScheduler, self).state_dict(destination)
        # TODO
        #destination['searcher'] = pickle.dumps(self.searcher)
        if self.visualizer == 'mxboard' or self.visualizer == 'tensorboard':
            destination['visualizer'] = json.dumps(self.mxboard._scalar_dict)
        return destination

    def load_state_dict(self, state_dict):
        super(DistributedFIFOScheduler, self).load_state_dict(state_dict)
        # TODO
        #self.searcher = pickle.loads(state_dict['searcher'])
        if self.visualizer == 'mxboard' or self.visualizer == 'tensorboard':
            self.mxboard._scalar_dict = json.loads(state_dict['visualizer'])
        logger.debug('Loading Searcher State {}'.format(self.searcher))
