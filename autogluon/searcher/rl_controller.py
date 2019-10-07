import pickle
import mxnet as mx
import mxnet.gluon.nn as nn
import mxnet.ndarray as F
from ..core.space import *
from .searcher import BaseSearcher
from ..utils import keydefaultdict, update_params
from collections import OrderedDict

__all__ = ['RLSearcher', 'LSTMController']

class RLSearcher(BaseSearcher):
    """Reinforcement Learning Searcher for ConfigSpace

    Args:
        configspace: ConfigSpace.ConfigurationSpace
            The configuration space to sample from. It contains the full
            specification of the Hyperparameters with their priors

    Example:
        >>> searcher = RLSearcher(cs)
        >>> searcher.get_config()
    """
    def __init__(self, kwspaces, ctx=mx.cpu(), controller_type='rl'):
        self._results = OrderedDict()
        self._best_state_path = None
        if controller_type == 'rl':
            self.controller = LSTMController(kwspaces, ctx=ctx)
        else:
            raise NotImplemented

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' +  \
            'Number of Trials: {}.'.format(len(self._results)) + \
            'Best Config: {}'.format(self.get_best_config()) + \
            'Best Reward: {}'.format(self.get_best_reward()) + \
            ')'
        return reprstr

    def get_config(self):
        return self.controller.sample()[0]

    def state_dict(self, destination=None):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination['results'] = pickle.dumps(self._results)
        destination['controller_params'] = pickle.dumps(self.controller.collect_params())
        return destination

    def load_state_dict(self, state_dict):
        self._results=pickle.loads(state_dict['results'])
        update_params(self.controller, pickle.loads(state_dict['controller_params']))


# Reference: https://github.com/carpedm20/ENAS-pytorch/
class LSTMController(mx.gluon.Block):
    def __init__(self, kwspaces, softmax_temperature=1.0, hidden_size=100,
                 ctx=mx.cpu()):
        super().__init__()
        self.softmax_temperature = softmax_temperature
        self.spaces = list(kwspaces.items())
        self.hidden_size = hidden_size
        self.context = ctx

        # only support List space for now
        self.num_tokens = []
        for _, space in self.spaces:
            assert isinstance(space, List)
            self.num_tokens.append(len(space))
        num_total_tokens = sum(self.num_tokens)

        # controller lstm
        self.encoder = nn.Embedding(num_total_tokens, hidden_size)
        self.lstm = mx.gluon.rnn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        self.decoders = nn.Sequential()
        for idx, size in enumerate(self.num_tokens):
            decoder = nn.Dense(in_units=hidden_size, units=size)
            self.decoders.add(decoder)

        def _init_hidden(batch_size):
            #print('batch_size, hidden_size', batch_size, hidden_size)
            zeros = mx.nd.zeros((batch_size, hidden_size), ctx=self.context)
            return zeros, zeros.copy()

        def _get_default_hidden(key):
            #print('key, hidden_size', key, hidden_size)
            return mx.nd.zeros((key, hidden_size), ctx=self.context)
        
        self.static_init_hidden = keydefaultdict(_init_hidden)
        self.static_inputs = keydefaultdict(_get_default_hidden)

    def initialize(self, ctx=[mx.cpu()], *args, **kwargs):
        self.context = ctx[0]
        super().initialize(ctx=ctx, *args, **kwargs)

    def forward(self, inputs, hidden, block_idx, is_embed):
        if not is_embed:
            embed = self.encoder(inputs)
        else:
            embed = inputs
        _, (hx, cx) = self.lstm(embed, hidden)

        logits = self.decoders[block_idx](hx)
        logits = logits / self.softmax_temperature

        return logits, (hx, cx)

    def sample(self, batch_size=1, with_details=False):
        """
        Return:
            configs (list of dict): list of configurations
        """
        inputs = self.static_inputs[batch_size]
        hidden = self.static_init_hidden[batch_size]

        actions = []
        entropies = []
        log_probs = []

        for block_idx in range(len(self.num_tokens)):
            logits, hidden = self.forward(inputs, hidden,
                                          block_idx, is_embed=(block_idx==0))

            probs = F.softmax(logits, axis=-1)
            log_prob = F.log_softmax(logits, axis=-1)
            entropy = -(log_prob * probs).sum(1, keepdims=False)

            action = mx.random.multinomial(probs, 1)
            ind = mx.nd.stack(mx.nd.arange(probs.shape[0], ctx=action.context),
                              action.astype('float32'))
            selected_log_prob = F.gather_nd(log_prob, ind)

            actions.append(action[:, 0])
            entropies.append(entropy)
            log_probs.append(selected_log_prob)

            # why add some constant?
            inputs = action[:, 0] #+ sum(self.num_tokens[:block_idx])
            inputs.detach()

        configs = []
        for idx in range(batch_size):
            config = {}
            for i, action in enumerate(actions):
                choice = action[idx].asscalar()
                k, space = self.spaces[i]
                config[k] = choice#space[choice]
            configs.append(config)

        if with_details:
            return configs, F.stack(*log_probs, axis=1), F.stack(*entropies, axis=1)
        else:
            return configs
