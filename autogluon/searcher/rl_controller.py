import mxnet as mx
import mxnet.gluon.nn as nn
import mxnet.ndarray as F
from ..basic.space import *
from .searcher import BaseSearcher
from ..utils import keydefaultdict

__all__ = ['RLSearcher', 'Controller']

class RLSearcher(BaseSearcher):
    """Random sampling Searcher for ConfigSpace

    Args:
        configspace: ConfigSpace.ConfigurationSpace
            The configuration space to sample from. It contains the full
            specification of the Hyperparameters with their priors

    Example:
        >>> searcher = RLSearcher(cs)
        >>> searcher.get_config()
    """
    def __init__(self, **kwspaces):
        #self.configspace = configspace
        self._results = OrderedDict()
        self._best_state_path = None
        self.controller = Controller(kwspaces)

    def get_config(self):
        """Function to sample a new configuration
        This function is called inside Hyperband to query a new configuration

        Args:
            returns: (config, info_dict)
                must return a valid configuration and a (possibly empty) info dict
        """
        while pickle.dumps(new_config) in self._results.keys():
            pass

    def update(self, *args, **kwargs):
        """Update the searcher with the newest metric report
        """
        super(RLSearcher, self).update(*args, **kwargs)

    #def train_controller(self):
    #    model = self.controller

    #    avg_reward_base = None
    #    baseline = None
    #    adv_history = []
    #    entropy_history = []
    #    reward_history = []

    #    hidden = self.shared.init_hidden(self.args.batch_size)
    #    total_loss = 0
    #    valid_idx = 0
    #    for step in range(self.args.controller_max_step):
    #        # sample models
    #        dags, log_probs, entropies = self.controller.sample(
    #            with_details=True)

    #        # calculate reward
    #        np_entropies = entropies.data.cpu().numpy()
    #        # NOTE(brendan): No gradients should be backpropagated to the
    #        # shared model during controller training, obviously.
    #        with _get_no_grad_ctx_mgr():
    #            rewards, hidden = self.get_reward(dags,
    #                                              np_entropies,
    #                                              hidden,
    #                                              valid_idx)

    #        # discount
    #        if 1 > self.args.discount > 0:
    #            rewards = discount(rewards, self.args.discount)

    #        reward_history.extend(rewards)
    #        entropy_history.extend(np_entropies)

    #        # moving average baseline
    #        if baseline is None:
    #            baseline = rewards
    #        else:
    #            decay = self.args.ema_baseline_decay
    #            baseline = decay * baseline + (1 - decay) * rewards

    #        adv = rewards - baseline
    #        adv_history.extend(adv)

    #        # policy loss
    #        loss = -log_probs*utils.get_variable(adv,
    #                                             self.cuda,
    #                                             requires_grad=False)
    #        if self.args.entropy_mode == 'regularizer':
    #            loss -= self.args.entropy_coeff * entropies

    #        loss = loss.sum()  # or loss.mean()

    #        # update
    #        self.controller_optim.zero_grad()
    #        loss.backward()

    #        if self.args.controller_grad_clip > 0:
    #            torch.nn.utils.clip_grad_norm(model.parameters(),
    #                                          self.args.controller_grad_clip)
    #        self.controller_optim.step()

    #        total_loss += utils.to_item(loss.data)

    #        if ((step % self.args.log_step) == 0) and (step > 0):
    #            self._summarize_controller_train(total_loss,
    #                                             adv_history,
    #                                             entropy_history,
    #                                             reward_history,
    #                                             avg_reward_base,
    #                                             dags)

    #            reward_history, adv_history, entropy_history = [], [], []
    #            total_loss = 0

    #        self.controller_step += 1

    #        prev_valid_idx = valid_idx
    #        valid_idx = ((valid_idx + self.max_length) %
    #                     (self.valid_data.size(0) - 1))
    #        # NOTE(brendan): Whenever we wrap around to the beginning of the
    #        # validation data, we reset the hidden states.
    #        if prev_valid_idx > valid_idx:
    #            hidden = self.shared.init_hidden(self.args.batch_size)


    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' +  \
            'Number of Trials: {}.'.format(len(self._results)) + \
            'Best Config: {}'.format(self.get_best_config()) + \
            'Best Reward: {}'.format(self.get_best_reward()) + \
            ')'
        return reprstr

# Reference: https://github.com/carpedm20/ENAS-pytorch/
class Controller(mx.gluon.Block):
    def __init__(self, kwspaces, softmax_temperature=1, controller_hid=100,
                 ctx=mx.cpu()):
        super().__init__()
        self.softmax_temperature = softmax_temperature
        self.spaces = list(kwspaces.items())
        self.controller_hid = controller_hid
        self.context = ctx

        # only support discrete space for now
        self.num_tokens = []
        for _, space in self.spaces:
            assert isinstance(space, ListSpace)
            self.num_tokens.append(len(space))
        num_total_tokens = sum(self.num_tokens)

        # controller lstm
        self.encoder = nn.Embedding(num_total_tokens,
                                    controller_hid)
        self.lstm = mx.gluon.rnn.LSTMCell(input_size=controller_hid,
                                          hidden_size=controller_hid)
        self.decoders = nn.Sequential()
        for idx, size in enumerate(self.num_tokens):
            decoder = nn.Dense(in_units=controller_hid, units=size)
            self.decoders.add(decoder)

        def _init_hidden(batch_size):
            zeros = mx.nd.zeros((batch_size, controller_hid), ctx=self.context)
            return zeros, zeros.copy()

        def _get_default_hidden(key):
            return mx.nd.zeros((key, controller_hid), ctx=self.context)
        
        self.static_init_hidden = keydefaultdict(_init_hidden)
        self.static_inputs = keydefaultdict(_get_default_hidden)

    def forward(self, inputs, hidden, block_idx, is_embed):
        if not is_embed:
            embed = self.encoder(inputs)
        else:
            embed = inputs
        print('embed', embed)
        print('hidden', hidden)
        _, (hx, cx) = self.lstm(embed, hidden)

        logits = self.decoders[block_idx](hx)
        logits /= self.softmax_temperature

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
            logits, hidden = self.forward(inputs,
                                          hidden,
                                          block_idx,
                                          is_embed=(block_idx==0))

            probs = F.softmax(logits, axis=-1)
            log_prob = F.log_softmax(logits, axis=-1)
            entropy = -(log_prob * probs).sum(1, keepdims=False)

            action = mx.random.multinomial(probs, 1)
            ind = mx.nd.stack(mx.nd.arange(probs.shape[0], ctx=action.context), action.astype('float32'))
            selected_log_prob = F.gather_nd(log_prob, ind)

            actions.append(action[:, 0])
            entropies.append(entropy)
            log_probs.append(selected_log_prob)

            # why add some constant?
            inputs = action[:, 0] + sum(self.num_tokens[:block_idx])
            inputs.detach()

        configs = []
        for idx in range(batch_size):
            config = {}
            for i, action in enumerate(actions):
                choice = action[idx].asscalar()
                k, space = self.spaces[i]
                config[k] = space[choice]
            configs.append(config)

        if with_details:
            return configs, F.concat(*log_probs, dim=0), F.concat(*entropies, dim=0)
        else:
            return configs
