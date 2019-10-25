import os
import pickle
import logging
from collections import OrderedDict

import mxnet as mx

from ...searcher import RLSearcher
from ...scheduler.resource import get_gpu_count, get_cpu_count
from ...task.image_classification.dataset import get_built_in_dataset
from ...utils import (mkdir, save, load, update_params, collect_params, DataLoader, in_ipynb)
from .enas_utils import *

if in_ipynb():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

__all__ = ['ENAS_Scheduler']

logger = logging.getLogger(__name__)

IMAGENET_TRAINING_SAMPLES = 1281167

class ENAS_Scheduler(object):
    """ENAS Scheduler, which automatically creates LSTM controller based on the search spaces.
    """
    def __init__(self, supernet, train_set='imagenet', val_set=None,
                 train_fn=default_train_fn, eval_fn=default_val_fn,
                 train_args={}, val_args={}, reward_fn= default_reward_fn,
                 num_gpus=get_gpu_count(),
                 num_cpu=get_cpu_count(), batch_size=256,
                 epochs=120, warmup_epochs=5,
                 controller_lr=0.01, controller_type='lstm',
                 controller_batch_size=10, ema_baseline_decay=0.95,
                 update_arch_frequency=20, checkname='./enas/checkpoint.ag',
                 plot_frequency=0, **kwargs):
        ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu(0)]
        supernet.collect_params().reset_ctx(ctx)
        supernet.hybridize()
        self.supernet = supernet
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.reward_fn = reward_fn
        dataset_name = train_set
        self.checkname = checkname
        self.plot_frequency = plot_frequency
        if isinstance(train_set, str):
            train_set = get_built_in_dataset(dataset_name, train=True, batch_size=batch_size, shuffle=True).init()
            val_set = get_built_in_dataset(dataset_name, train=False, batch_size=batch_size, shuffle=True).init()
        if isinstance(train_set, gluon.data.Dataset):
            self.train_data = DataLoader(
                    train_set, batch_size=batch_size, shuffle=True,
                    last_batch="discard", num_workers=num_cpu)
            # very important, make shuffle for training contoller
            self.val_data = DataLoader(
                    val_set, batch_size=batch_size, shuffle=True,
                    num_workers=num_cpu, prefetch=0, sample_times=controller_batch_size)
        else:
            self.train_data = train_set
            self.val_data = val_set
        iters_per_epoch = len(train_set) if hasattr(train_set, '__len__') else IMAGENET_TRAINING_SAMPLES // batch_size
        self.train_args = init_default_train_args(batch_size, self.supernet, epochs, iters_per_epoch) \
                if len(train_args) == 0 else train_args
        self.val_args = val_args
        self.val_args['ctx'] = ctx
        self.val_args['batch_fn'] = imagenet_batch_fn if dataset_name == 'imagenet' else default_batch_fn
        self.train_args['ctx'] = ctx
        self.train_args['batch_fn'] = imagenet_batch_fn if dataset_name == 'imagenet' else default_batch_fn
        self.ctx = ctx

        self.epochs = epochs
        self.baseline = None
        self.warmup_epochs = warmup_epochs
        # create RL searcher/controller
        self.ema_decay = ema_baseline_decay
        self.searcher = RLSearcher(self.supernet.kwspaces, controller_type=controller_type)
        # controller setup
        self.controller = self.searcher.controller
        controller_resource = mx.gpu(0) if get_gpu_count() > 0 else mx.cpu(0)
        self.controller.collect_params().reset_ctx([controller_resource])
        self.controller.context = controller_resource
        self.controller_optimizer = mx.gluon.Trainer(
                self.controller.collect_params(), 'adam',
                optimizer_params={'learning_rate': controller_lr*controller_batch_size})
        self.update_arch_frequency = update_arch_frequency
        self.controller_batch_size = controller_batch_size
        self.val_acc = 0

    def run(self):
        tq = tqdm(range(self.epochs))
        for epoch in tq:
            # for recordio data
            if hasattr(self.train_data, 'reset'): self.train_data.reset()
            tbar = tqdm(enumerate(self.train_data))
            for i, batch in tbar:
                # sample network configuration
                config = self.controller.sample()[0]
                self.supernet.sample(**config)
                self.train_fn(self.supernet, batch, **self.train_args)
                if epoch >= self.warmup_epochs and (i % self.update_arch_frequency) == 0:
                    self.train_controller()
                if self.plot_frequency > 0 and i % self.plot_frequency == 0:
                    from IPython.display import SVG, display, clear_output
                    clear_output(wait=True)
                    graph = self.supernet.graph
                    graph.attr(rankdir='LR', size='8,3')
                    display(SVG(graph._repr_svg_()))
                tbar.set_description('epoch {}, iter {}, val_acc: {}, avg reward: {}' \
                        .format(epoch, i, self.val_acc, self.baseline))
            self.validation()
            self.save()
            tq.set_description('epoch {}, val_acc: {}, avg reward: {}' \
                        .format(epoch, self.val_acc, self.baseline))

    def validation(self):
        if hasattr(self.val_data, 'reset'): self.val_data.reset()
        sum_rewards = 0
        # data iter
        tbar = tqdm(enumerate(self.val_data))
        # update network arc
        config = self.controller.inference()
        self.supernet.sample(**config)
        for i, batch in tbar:
            reward = self.eval_fn(self.supernet, batch, **self.val_args)
            sum_rewards += reward
            tbar.set_description('Acc: {}'.format(sum_rewards/(i+1)))

        self.val_acc = sum_rewards / (i+1)

    def train_controller(self):
        """Run multiple number of trials
        """
        decay = self.ema_decay
        if hasattr(self.val_data, 'reset'): self.val_data.reset()
        # update 
        for i, batch in enumerate(self.val_data):
            if i >= self.controller_batch_size: break
            with mx.autograd.record():
                # sample controller_batch_size number of configurations
                configs, log_probs, entropies = self.controller.sample(batch_size=1, with_details=True)
                # schedule the training tasks and gather the reward
                self.supernet.sample(**configs[0])
                metric = self.eval_fn(self.supernet, batch, **self.val_args)
                reward = self.reward_fn(metric, self.supernet)
                self.baseline = reward if not self.baseline else self.baseline
                # substract baseline
                avg_rewards = mx.nd.array([reward - self.baseline],
                                          ctx=self.controller.context)
                # EMA baseline
                self.baseline = decay * self.baseline + (1 - decay) * reward
                # negative policy gradient
                log_probs = log_probs.sum(axis=1)
                loss = - log_probs * avg_rewards
                loss = loss.sum()

        # update
        loss.backward()
        self.controller_optimizer.step(self.controller_batch_size)
        logger.debug('controller loss: {}'.format(loss.asscalar()))

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
        return destination

    def load_state_dict(self, state_dict):
        update_params(self.supernet, state_dict['supernet_params'])
        update_params(self.controller, state_dict['controller_params'])
