import os
import pickle
import logging
from collections import OrderedDict
from multiprocessing.pool import ThreadPool

import mxnet as mx

from ...searcher import RLSearcher
from ...scheduler.resource import get_gpu_count, get_cpu_count
from ...task.image_classification.dataset import get_built_in_dataset
from ...task.image_classification.utils import *
from ...utils import (mkdir, save, load, update_params, collect_params, DataLoader, tqdm, in_ipynb)
from .enas_utils import *

__all__ = ['ENAS_Scheduler']

logger = logging.getLogger(__name__)

IMAGENET_TRAINING_SAMPLES = 1281167

class ENAS_Scheduler(object):
    """ENAS Scheduler, which automatically creates LSTM controller based on the search spaces.
    """
    def __init__(self, supernet, train_set='imagenet', val_set=None,
                 train_fn=default_train_fn, eval_fn=default_val_fn,
                 train_args={}, val_args={}, reward_fn= default_reward_fn,
                 num_gpus=0, num_cpus=4,
                 batch_size=256, epochs=120, warmup_epochs=5,
                 controller_lr=1e-3, controller_type='lstm',
                 controller_batch_size=10, ema_baseline_decay=0.95,
                 update_arch_frequency=20, checkname='./enas/checkpoint.ag',
                 plot_frequency=0, **kwargs):
        num_cpus = get_cpu_count() if num_cpus > get_cpu_count() else num_cpus
        num_gpus = get_gpu_count() if num_gpus > get_gpu_count() else num_gpus
        self.supernet = supernet
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.reward_fn = reward_fn
        self.checkname = checkname
        self.plot_frequency = plot_frequency
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.controller_batch_size = controller_batch_size
        kwspaces = self.supernet.kwspaces

        self.initialize_miscs(train_set, val_set, batch_size, num_cpus, num_gpus,
                              train_args, val_args)

        # create RL searcher/controller
        self.baseline = None
        self.ema_decay = ema_baseline_decay
        self.searcher = RLSearcher(
            kwspaces, controller_type=controller_type, prefetch=4,
            num_workers=4)
        # controller setup
        self.controller = self.searcher.controller
        self.controller_optimizer = mx.gluon.Trainer(
                self.controller.collect_params(), 'adam',
                optimizer_params={'learning_rate': controller_lr})
        self.update_arch_frequency = update_arch_frequency
        self.val_acc = 0
        # async controller sample
        self._worker_pool = ThreadPool(2)
        self._data_buffer = {}
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._timeout = 20
        # logging history
        self.training_history = []
        self._prefetch_controller()

    def initialize_miscs(self, train_set, val_set, batch_size, num_cpus, num_gpus,
                         train_args, val_args):
        """Initialize framework related miscs, such as train/val data and train/val
        function arguments.
        """
        ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu(0)]
        self.supernet.collect_params().reset_ctx(ctx)
        self.supernet.hybridize()
        dataset_name = train_set

        if isinstance(train_set, str):
            train_set = get_built_in_dataset(dataset_name, train=True, batch_size=batch_size,
                                             num_workers=num_cpus, shuffle=True)
            val_set = get_built_in_dataset(dataset_name, train=False, batch_size=batch_size,
                                           num_workers=num_cpus, shuffle=True)
        if isinstance(train_set, gluon.data.Dataset):
            self.train_data = DataLoader(
                    train_set, batch_size=batch_size, shuffle=True,
                    last_batch="discard", num_workers=num_cpus)
            # very important, make shuffle for training contoller
            self.val_data = DataLoader(
                    val_set, batch_size=batch_size, shuffle=True,
                    num_workers=num_cpus, prefetch=0, sample_times=self.controller_batch_size)
        else:
            self.train_data = train_set
            self.val_data = val_set
        iters_per_epoch = len(self.train_data) if hasattr(self.train_data, '__len__') else \
                IMAGENET_TRAINING_SAMPLES // batch_size
        self.train_args = init_default_train_args(batch_size, self.supernet, self.epochs, iters_per_epoch) \
                if len(train_args) == 0 else train_args
        self.val_args = val_args
        self.val_args['ctx'] = ctx
        self.val_args['batch_fn'] = imagenet_batch_fn if dataset_name == 'imagenet' else default_batch_fn
        self.train_args['ctx'] = ctx
        self.train_args['batch_fn'] = imagenet_batch_fn if dataset_name == 'imagenet' else default_batch_fn
        self.ctx = ctx

    def run(self):
        tq = tqdm(range(self.epochs))
        for epoch in tq:
            # for recordio data
            if hasattr(self.train_data, 'reset'): self.train_data.reset()
            tbar = tqdm(self.train_data)
            idx = 0
            for batch in tbar:
                # sample network configuration
                config = self.controller.pre_sample()[0]
                self.supernet.sample(**config)
                # self.train_fn(self.supernet, batch, **self.train_args)
                self.train_fn(epoch, self.epochs, self.supernet, batch, **self.train_args)
                mx.nd.waitall()
                if epoch >= self.warmup_epochs and (idx % self.update_arch_frequency) == 0:
                    self.train_controller()
                if self.plot_frequency > 0 and idx % self.plot_frequency == 0 and in_ipynb():
                    graph = self.supernet.graph
                    graph.attr(rankdir='LR', size='8,3')
                    tbar.set_svg(graph._repr_svg_())
                if self.baseline:
                    tbar.set_description('avg reward: {:.2f}'.format(self.baseline))
                idx += 1
            self.validation()
            self.save()
            msg = 'epoch {}, val_acc: {:.2f}'.format(epoch, self.val_acc)
            if self.baseline:
                msg += ', avg reward: {:.2f}'.format(self.baseline)
            tq.set_description(msg)

    def validation(self):
        if hasattr(self.val_data, 'reset'): self.val_data.reset()
        # data iter, avoid memory leak
        it = iter(self.val_data)
        if hasattr(it, 'reset_sample_times'): it.reset_sample_times()
        tbar = tqdm(it)
        # update network arc
        config = self.controller.inference()
        self.supernet.sample(**config)
        metric = mx.metric.Accuracy()
        for batch in tbar:
            self.eval_fn(self.supernet, batch, metric=metric, **self.val_args)
            reward = metric.get()[1]
            tbar.set_description('Val Acc: {}'.format(reward))

        self.val_acc = reward
        self.training_history.append(reward)

    def _sample_controller(self):
        assert self._rcvd_idx < self._sent_idx, "rcvd_idx must be smaller than sent_idx"
        try:
            ret = self._data_buffer.pop(self._rcvd_idx)
            self._rcvd_idx += 1
            return  ret.get(timeout=self._timeout)
        except Exception:
            self._worker_pool.terminate()
            raise

    def _prefetch_controller(self):
        async_ret = self._worker_pool.apply_async(self._async_sample, ())
        self._data_buffer[self._sent_idx] = async_ret
        self._sent_idx += 1

    def _async_sample(self):
        with mx.autograd.record():
            # sample controller_batch_size number of configurations
            configs, log_probs, entropies = self.controller.sample(batch_size=self.controller_batch_size,
                                                                   with_details=True)
        return configs, log_probs, entropies

    def train_controller(self):
        """Run multiple number of trials
        """
        decay = self.ema_decay
        if hasattr(self.val_data, 'reset'): self.val_data.reset()
        # update 
        metric = mx.metric.Accuracy()
        with mx.autograd.record():
            # sample controller_batch_size number of configurations
            configs, log_probs, entropies = self._sample_controller()
            for i, batch in enumerate(self.val_data):
                if i >= self.controller_batch_size: break
                self.supernet.sample(**configs[i])
                # schedule the training tasks and gather the reward
                metric.reset()
                self.eval_fn(self.supernet, batch, metric=metric, **self.val_args)
                reward = metric.get()[1]
                reward = self.reward_fn(reward, self.supernet)
                self.baseline = reward if not self.baseline else self.baseline
                # substract baseline
                avg_rewards = mx.nd.array([reward - self.baseline],
                                          ctx=self.controller.context)
                # EMA baseline
                self.baseline = decay * self.baseline + (1 - decay) * reward
                # negative policy gradient
                log_prob = log_probs[i]
                log_prob = log_prob.sum()
                loss = - log_prob * avg_rewards
                loss = loss.sum()

        # update
        loss.backward()
        self.controller_optimizer.step(self.controller_batch_size)
        self._prefetch_controller()

    def load(self, checkname=None):
        checkname = checkname if checkname else self.checkname
        state_dict = load(checkname)
        self.load_state_dict(state_dict)

    def save(self, checkname=None):
        checkname = checkname if checkname else self.checkname
        mkdir(os.path.dirname(checkname))
        save(self.state_dict(), checkname)

    def state_dict(self, destination=None):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination['supernet_params'] = collect_params(self.supernet)
        destination['controller_params'] = collect_params(self.controller)
        destination['training_history'] = self.training_history
        return destination

    def load_state_dict(self, state_dict):
        update_params(self.supernet, state_dict['supernet_params'], ctx=self.ctx)
        update_params(self.controller, state_dict['controller_params'], ctx=self.controller.context)
        self.training_history = state_dict['training_history']
