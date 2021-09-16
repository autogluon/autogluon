import mxnet as mx
import mxnet.gluon.nn as nn
import multiprocessing
from multiprocessing.pool import ThreadPool

from autogluon.core.utils import keydefaultdict
from autogluon.core.space import Categorical


class BaseController(mx.gluon.Block):
    """
    BaseController subclasses are used in RLSearcher, which is the searcher
    for RLScheduler.
    """
    def __init__(self, prefetch=4, num_workers=4, timeout=20, **kwargs):
        super().__init__()
        # manager = multiprocessing.Manager()
        self._data_buffer = {}  # manager.dict()
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._num_workers = num_workers
        self._worker_pool = ThreadPool(self._num_workers)
        self._timeout = timeout
        self._nprefetch = prefetch

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def _prefetch(self):
        async_ret = self._worker_pool.apply_async(self.sample, ())
        self._data_buffer[self._sent_idx] = async_ret
        self._sent_idx += 1

    def initialize(self, ctx=mx.cpu(), *args, **kwargs):
        self.context = ctx[0] if isinstance(ctx, (list, tuple)) else ctx
        super().initialize(ctx=ctx, *args, **kwargs)

    def pre_sample(self):
        if self._rcvd_idx == self._sent_idx:
            self._prefetch()
        self._prefetch()
        assert self._rcvd_idx < self._sent_idx, "rcvd_idx must be smaller than sent_idx"
        try:
            ret = self._data_buffer.pop(self._rcvd_idx)
            self._rcvd_idx += 1
            return ret.get(timeout=self._timeout)
        except multiprocessing.context.TimeoutError:
            msg = '''Worker timed out after {} seconds. This might be caused by \n
            - Slow transform. Please increase timeout to allow slower data loading in each worker.
            '''.format(self._timeout)
            print(msg)
            raise
        except Exception:
            self._worker_pool.terminate()
            raise


# Reference: https://github.com/carpedm20/ENAS-pytorch/
class LSTMController(BaseController):
    def __init__(self, kwspaces, softmax_temperature=1.0, hidden_size=100,
                 ctx=mx.cpu(), **kwargs):
        super().__init__(**kwargs)
        self.softmax_temperature = softmax_temperature
        self.spaces = list(kwspaces.items())
        self.hidden_size = hidden_size
        self.context = ctx

        # only support Categorical space for now
        self.num_tokens = []
        for _, space in self.spaces:
            assert isinstance(space, Categorical)
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
            zeros = mx.nd.zeros((batch_size, hidden_size), ctx=self.context)
            return zeros, zeros.copy()

        def _get_default_hidden(key):
            return mx.nd.zeros((key, hidden_size), ctx=self.context)

        self.static_init_hidden = keydefaultdict(_init_hidden)
        self.static_inputs = keydefaultdict(_get_default_hidden)

    def forward(self, inputs, hidden, block_idx, is_embed):
        if not is_embed:
            embed = self.encoder(inputs)
        else:
            embed = inputs
        _, (hx, cx) = self.lstm(embed, hidden)

        logits = self.decoders[block_idx](hx)
        logits = logits / self.softmax_temperature

        return logits, (hx, cx)

    def inference(self):
        inputs = self.static_inputs[1]
        hidden = self.static_init_hidden[1]
        actions = []
        for block_idx in range(len(self.num_tokens)):
            logits, hidden = self.forward(inputs, hidden,
                                          block_idx, is_embed=(block_idx == 0))
            probs = mx.nd.softmax(logits, axis=-1)
            action = mx.nd.argmax(probs, 1)
            actions.append(action)
            inputs = action + sum(self.num_tokens[:block_idx])
            inputs.detach()

        config = {}
        for i, action in enumerate(actions):
            choice = action.asscalar()
            k, space = self.spaces[i]
            config[k] = int(choice)

        return config

    def sample(self, batch_size=1, with_details=False, with_entropy=False):
        """
        Returns
        -------
        configs : list of dict
            list of configurations
        """
        inputs = self.static_inputs[batch_size]
        hidden = self.static_init_hidden[batch_size]

        actions = []
        entropies = []
        log_probs = []

        for idx in range(len(self.num_tokens)):
            logits, hidden = self.forward(inputs, hidden,
                                          idx, is_embed=(idx == 0))

            probs = mx.nd.softmax(logits, axis=-1)
            log_prob = mx.nd.log_softmax(logits, axis=-1)
            entropy = -(log_prob * probs).sum(1, keepdims=False) if with_entropy else None

            action = mx.random.multinomial(probs, 1)
            ind = mx.nd.stack(mx.nd.arange(probs.shape[0], ctx=action.context),
                              action.astype('float32'))
            selected_log_prob = mx.nd.gather_nd(log_prob, ind)

            actions.append(action[:, 0])
            entropies.append(entropy)
            log_probs.append(selected_log_prob)

            inputs = action[:, 0] + sum(self.num_tokens[:idx])
            inputs.detach()

        configs = []
        for idx in range(batch_size):
            config = {}
            for i, action in enumerate(actions):
                choice = action[idx].asscalar()
                k, space = self.spaces[i]
                config[k] = int(choice)
            configs.append(config)

        if with_details:
            entropies = mx.nd.stack(*entropies, axis=1) if with_entropy else entropies
            return configs, mx.nd.stack(*log_probs, axis=1), entropies
        else:
            return configs


class Alpha(mx.gluon.Block):
    def __init__(self, shape):
        super().__init__()
        self.weight = self.params.get('weight', shape=shape)

    def forward(self, batch_size):
        return self.weight.data().expand_dims(0).repeat(batch_size, axis=0)


class AttenController(BaseController):
    def __init__(self, kwspaces, softmax_temperature=1.0, hidden_size=100,
                 ctx=mx.cpu(), **kwargs):
        super().__init__(**kwargs)
        self.softmax_temperature = softmax_temperature
        self.spaces = list(kwspaces.items())
        self.context = ctx

        # only support Categorical space for now
        self.num_tokens = []
        for _, space in self.spaces:
            assert isinstance(space, Categorical)
            self.num_tokens.append(len(space))
        self.num_total_tokens = sum(self.num_tokens)
        self.hidden_size = hidden_size

        self.embedding = Alpha((self.num_total_tokens, hidden_size))
        self.querry = nn.Dense(hidden_size, in_units=hidden_size)
        self.key = nn.Dense(hidden_size, in_units=hidden_size)
        self.value = nn.Dense(1, in_units=hidden_size)

    def inference(self):
        # self-attention
        x = self.embedding(1).reshape(-3, 0)  # .squeeze() # b x action x h
        kshape = (1, self.num_total_tokens, self.hidden_size)
        vshape = (1, self.num_total_tokens, 1)
        querry = self.querry(x).reshape(*kshape)  # b x actions x h
        key = self.key(x).reshape(*kshape)  # b x actions x h
        value = self.value(x).reshape(*vshape)  # b x actions x 1
        atten = mx.nd.linalg_gemm2(querry, key, transpose_b=True).softmax(axis=1)
        alphas = mx.nd.linalg_gemm2(atten, value).squeeze(axis=-1)

        actions = []
        for idx in range(len(self.num_tokens)):
            i0 = sum(self.num_tokens[:idx])
            i1 = sum(self.num_tokens[:idx + 1])
            logits = alphas[:, i0: i1]
            probs = mx.nd.softmax(logits, axis=-1)
            action = mx.nd.argmax(probs, 1)
            actions.append(action)

        config = {}
        for i, action in enumerate(actions):
            choice = action.asscalar()
            k, space = self.spaces[i]
            config[k] = int(choice)

        return config

    def sample(self, batch_size=1, with_details=False, with_entropy=False):
        # self-attention
        x = self.embedding(batch_size).reshape(-3, 0)  # .squeeze() # b x action x h
        kshape = (batch_size, self.num_total_tokens, self.hidden_size)
        vshape = (batch_size, self.num_total_tokens, 1)
        querry = self.querry(x).reshape(*kshape)  # b x actions x h
        key = self.key(x).reshape(*kshape)  # b x actions x h
        value = self.value(x).reshape(*vshape)  # b x actions x 1
        atten = mx.nd.linalg_gemm2(querry, key, transpose_b=True).softmax(axis=1)
        alphas = mx.nd.linalg_gemm2(atten, value).squeeze(axis=-1)

        actions = []
        entropies = []
        log_probs = []
        for idx in range(len(self.num_tokens)):
            i0 = sum(self.num_tokens[:idx])
            i1 = sum(self.num_tokens[:idx + 1])
            logits = alphas[:, i0: i1]

            probs = mx.nd.softmax(logits, axis=-1)
            log_prob = mx.nd.log_softmax(logits, axis=-1)

            entropy = -(log_prob * probs).sum(1, keepdims=False) if with_entropy else None

            action = mx.random.multinomial(probs, 1)
            ind = mx.nd.stack(mx.nd.arange(probs.shape[0], ctx=action.context),
                              action.astype('float32'))
            selected_log_prob = mx.nd.gather_nd(log_prob, ind)

            actions.append(action[:, 0])
            entropies.append(entropy)
            log_probs.append(selected_log_prob)

        configs = []
        for idx in range(batch_size):
            config = {}
            for i, action in enumerate(actions):
                choice = action[idx].asscalar()
                k, space = self.spaces[i]
                config[k] = int(choice)
            configs.append(config)

        if with_details:
            entropies = mx.nd.stack(*entropies, axis=1) if with_entropy else entropies
            return configs, mx.nd.stack(*log_probs, axis=1), entropies
        else:
            return configs


class AlphaController(BaseController):
    def __init__(self, kwspaces, softmax_temperature=1.0, ctx=mx.cpu(), **kwargs):
        super().__init__(**kwargs)
        self.softmax_temperature = softmax_temperature
        self.spaces = list(kwspaces.items())
        self.context = ctx

        # only support Categorical space for now
        self.num_tokens = []
        for _, space in self.spaces:
            assert isinstance(space, Categorical)
            self.num_tokens.append(len(space))

        # controller lstm
        self.decoders = nn.Sequential()
        for idx, size in enumerate(self.num_tokens):
            self.decoders.add(Alpha((size,)))

    def inference(self):
        actions = []

        for idx in range(len(self.num_tokens)):
            logits = self.decoders[idx](1)
            probs = mx.nd.softmax(logits, axis=-1)
            action = mx.nd.argmax(probs, 1)
            actions.append(action)

        config = {}
        for i, action in enumerate(actions):
            choice = action.asscalar()
            k, space = self.spaces[i]
            config[k] = int(choice)

        return config

    def sample(self, batch_size=1, with_details=False, with_entropy=False):
        actions = []
        entropies = []
        log_probs = []

        for idx in range(len(self.num_tokens)):
            logits = self.decoders[idx](batch_size)

            probs = mx.nd.softmax(logits, axis=-1)
            log_prob = mx.nd.log_softmax(logits, axis=-1)

            entropy = -(log_prob * probs).sum(1, keepdims=False) if with_entropy else None

            action = mx.random.multinomial(probs, 1)
            ind = mx.nd.stack(mx.nd.arange(probs.shape[0], ctx=action.context),
                              action.astype('float32'))
            selected_log_prob = mx.nd.gather_nd(log_prob, ind)

            actions.append(action[:, 0])
            entropies.append(entropy)
            log_probs.append(selected_log_prob)

        configs = []
        for idx in range(batch_size):
            config = {}
            for i, action in enumerate(actions):
                choice = action[idx].asscalar()
                k, space = self.spaces[i]
                config[k] = int(choice)
            configs.append(config)

        if with_details:
            entropies = mx.nd.stack(*entropies, axis=1) if with_entropy else entropies
            return configs, mx.nd.stack(*log_probs, axis=1), entropies
        else:
            return configs
