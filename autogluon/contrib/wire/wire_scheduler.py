import mxnet as mx

from ..enas import ENAS_Scheduler
from ...utils import in_ipynb
if in_ipynb():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

__all__ = ['Wire_Scheduler']

class Wire_Scheduler(ENAS_Scheduler):
    """ENAS Scheduler, which automatically creates LSTM controller based on the search spaces.
    """
    def __init__(self, supernet, *args, reward_fn=lambda metric, net: metric, **kwargs):
        super(Wire_Scheduler, self).__init__(supernet, *args, reward_fn=reward_fn, **kwargs)

    def run(self):
        tq = tqdm(range(self.epochs))
        for epoch in tq:
            # for recordio data
            if hasattr(self.train_data, 'reset'): self.train_data.reset()
            tbar = tqdm(enumerate(self.train_data))
            for i, batch in tbar:
                # sample network configuration
                while True:
                    config = self.controller.sample()[0]
                    if self.supernet.sample(**config): break
                self.train_fn(self.supernet, batch, **self.train_args)
                if epoch >= self.warmup_epochs and (i % self.update_arch_frequency) == 0:
                    self.train_controller()
                if self.plot_frequency > 0 and i % self.plot_frequency == 0:
                    from IPython.display import SVG, display, clear_output
                    clear_output(wait=True)
                    display(SVG(self.supernet.graph._repr_svg_()))
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
        if not self.supernet.sample(**config):
            while True:
                config = self.controller.sample()[0]
                if self.supernet.sample(**config): break
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
                count = 0
                while True:
                    configs, log_probs, entropies = self.controller.sample(batch_size=1, with_details=True)
                    if self.supernet.sample(**configs[0]): break
                    count += 1
                    #if count > 5: break
                #if count > 5:
                #    # penalize incorrect configurations
                #    reward = 0.0
                #else:
                # schedule the training tasks and gather the reward
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
