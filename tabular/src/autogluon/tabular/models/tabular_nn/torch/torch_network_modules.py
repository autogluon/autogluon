import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from ..utils.nn_architecture_utils import getEmbedSizes

logger = logging.getLogger(__name__)


class EmbedNet(nn.Module):
    def __init__(self):
        pass


class MultiQuantileNet(nn.Module):
    def __init__(self,
                 quantile_levels,
                 train_dataset=None,
                 params=None,
                 architecture_desc=None,
                 device=None):
        if (architecture_desc is None) and (train_dataset is None or params is None):
            raise ValueError("train_dataset, params cannot = None if architecture_desc=None")
        super(MultiQuantileNet, self).__init__()
        self.register_buffer('quantile_levels', torch.Tensor(quantile_levels).float().reshape(1, -1))
        self.device = torch.device('cpu') if device is None else device
        if architecture_desc is None:
            # adpatively specify network architecture based on training dataset
            self.from_logits = False
            self.has_vector_features = train_dataset.has_vector_features()
            self.has_embed_features = train_dataset.num_embed_features() > 0
            self.has_language_features = train_dataset.num_language_features() > 0
            if self.has_embed_features:
                num_categs_per_feature = train_dataset.getNumCategoriesEmbeddings()
                embed_dims = getEmbedSizes(train_dataset, params, num_categs_per_feature)
            if self.has_vector_features:
                vector_dims = train_dataset.data_list[train_dataset.vectordata_index].shape[-1]
        else:
            # ignore train_dataset, params, etc. Recreate architecture based on description:
            self.architecture_desc = architecture_desc
            self.has_vector_features = architecture_desc['has_vector_features']
            self.has_embed_features = architecture_desc['has_embed_features']
            self.has_language_features = architecture_desc['has_language_features']
            self.from_logits = architecture_desc['from_logits']
            params = architecture_desc['params']
            if self.has_embed_features:
                num_categs_per_feature = architecture_desc['num_categs_per_feature']
                embed_dims = architecture_desc['embed_dims']
            if self.has_vector_features:
                vector_dims = architecture_desc['vector_dims']
        # init input / output size
        input_size = 0
        output_size = self.quantile_levels.size()[-1]

        # define embedding layer:
        if self.has_embed_features:
            self.embed_blocks = nn.ModuleList()
            for i in range(len(num_categs_per_feature)):
                self.embed_blocks.append(nn.Embedding(num_embeddings=num_categs_per_feature[i],
                                                      embedding_dim=embed_dims[i]))
                input_size += embed_dims[i]

        # language block not supported
        if self.has_language_features:
            self.text_block = None
            raise NotImplementedError("text data cannot be handled")

        # update input size
        if self.has_vector_features:
            input_size += vector_dims

        # activation
        act_fn = nn.Identity()
        if params['activation'] == 'elu':
            act_fn = nn.ELU()
        elif params['activation'] == 'relu':
            act_fn = nn.ReLU()
        elif params['activation'] == 'tanh':
            act_fn = nn.Tanh()

        # layers
        layers = [nn.Linear(input_size, params['hidden_size']), act_fn]
        for _ in range(params['num_layers'] - 1):
            layers.append(nn.Dropout(params['dropout_prob']))
            layers.append(nn.Linear(params['hidden_size'], params['hidden_size']))
            layers.append(act_fn)
        layers.append(nn.Linear(params['hidden_size'], output_size))
        self.main_block = nn.Sequential(*layers)

        # set range for output
        y_range = params['y_range']  # Used specifically for regression. = None for classification.
        self.y_constraint = None  # determines if Y-predictions should be constrained
        if y_range is not None:
            if y_range[0] == -np.inf and y_range[1] == np.inf:
                self.y_constraint = None  # do not worry about Y-range in this case
            elif y_range[0] >= 0 and y_range[1] == np.inf:
                self.y_constraint = 'nonnegative'
            elif y_range[0] == -np.inf and y_range[1] <= 0:
                self.y_constraint = 'nonpositive'
            else:
                self.y_constraint = 'bounded'
            self.y_lower = y_range[0]
            self.y_upper = y_range[1]
            self.y_span = self.y_upper - self.y_lower

        # for huber loss
        self.alpha = params['alpha']

        if architecture_desc is None:  # Save Architecture description
            self.architecture_desc = {'has_vector_features': self.has_vector_features,
                                      'has_embed_features': self.has_embed_features,
                                      'has_language_features': self.has_language_features,
                                      'params': params,
                                      'from_logits': self.from_logits}
            if self.has_embed_features:
                self.architecture_desc['num_categs_per_feature'] = num_categs_per_feature
                self.architecture_desc['embed_dims'] = embed_dims
            if self.has_vector_features:
                self.architecture_desc['vector_dims'] = vector_dims

    def init_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, data_batch):
        input_data = []
        if self.has_vector_features:
            input_data.append(data_batch['vector'].to(self.device))
        if self.has_embed_features:
            embed_data = data_batch['embed']
            for i in range(len(self.embed_blocks)):
                input_data.append(self.embed_blocks[i](embed_data[i].to(self.device)))
        if self.has_language_features:
            pass

        if len(input_data) > 1:
            input_data = torch.cat(input_data, dim=1)
        else:
            input_data = input_data[0]

        output_data = self.main_block(input_data)

        # output with y-range
        if self.y_constraint is None:
            return output_data
        else:
            if self.y_constraint == 'nonnegative':
                return self.y_lower + torch.abs(output_data)
            elif self.y_constraint == 'nonpositive':
                return self.y_upper - torch.abs(output_data)
            else:
                return torch.sigmoid(output_data) * self.y_span + self.y_lower

    def monotonize(self, input_data):
        # number of quantiles
        num_quantiles = input_data.size()[-1]

        # split into below 50% and above 50%
        idx_50 = num_quantiles // 2

        # if a small number of quantiles are estimated or quantile levels are not centered at 0.5
        if num_quantiles < 3 or self.quantile_levels[0, idx_50] != 0.5:
            return input_data

        # below 50%
        below_50 = input_data[:, :(idx_50 + 1)].contiguous()
        below_50 = torch.flip(torch.cummin(torch.flip(below_50, [-1]), -1)[0], [-1])

        # above 50%
        above_50 = input_data[:, idx_50:].contiguous()
        above_50 = torch.cummax(above_50, -1)[0]

        # refined output
        ordered_data = torch.cat([below_50[:, :-1], above_50], -1)
        return ordered_data

    def huber_pinball_loss(self, input_data, target_data):
        error_data = target_data.contiguous().reshape(-1, 1) - input_data
        loss_data = torch.where(torch.abs(error_data) < self.alpha,
                                0.5 * error_data * error_data,
                                self.alpha * (torch.abs(error_data) - 0.5 * self.alpha))
        loss_data /= self.alpha

        scale = torch.where(error_data >= 0,
                            torch.ones_like(error_data) * self.quantile_levels,
                            torch.ones_like(error_data) * (1 - self.quantile_levels))
        loss_data *= scale
        return loss_data.mean()

    def margin_loss(self, input_data, margin_scale=0.0001):
        # number of samples
        batch_size, num_quantiles = input_data.size()

        # compute margin loss (batch_size x output_size(above) x output_size(below))
        error_data = input_data.unsqueeze(1) - input_data.unsqueeze(2)

        # margin data (num_quantiles x num_quantiles)
        margin_data = self.quantile_levels.permute(1, 0) - self.quantile_levels
        margin_data = torch.tril(margin_data, -1) * margin_scale

        # compute accumulated margin
        loss_data = torch.tril(error_data + margin_data, diagonal=-1)
        loss_data = loss_data.relu()
        loss_data = loss_data.sum() / float(batch_size * (num_quantiles * num_quantiles - num_quantiles) * 0.5)
        return loss_data

    def compute_loss(self,
                     data_batch,
                     weight=0.0,
                     margin=0.0):
        # train mode
        self.train()
        predict_data = self(data_batch)
        target_data = data_batch['label'].to(self.device)

        # get prediction and margin loss
        if margin > 0.0:
            m_loss = self.margin_loss(predict_data)
        else:
            m_loss = 0.0

        h_loss = (1 - weight) * self.huber_pinball_loss(self.monotonize(predict_data), target_data).mean()
        h_loss += weight * self.huber_pinball_loss(predict_data, target_data).mean()
        return h_loss + margin * m_loss

    def predict(self, input_data):
        self.eval()
        with torch.no_grad():
            predict_data = self(input_data)
            predict_data = self.monotonize(predict_data)
            return predict_data.data.cpu().numpy()
