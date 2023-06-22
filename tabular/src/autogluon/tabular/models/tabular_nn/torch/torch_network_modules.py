import logging

import numpy as np
import torch
import torch.nn as nn

from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REGRESSION, SOFTCLASS

from ..utils.nn_architecture_utils import get_embed_sizes

logger = logging.getLogger(__name__)


class EmbedNet(nn.Module):
    """
    y_range: Used specifically for regression. = None for classification.
    """

    def __init__(self, problem_type, num_net_outputs=None, quantile_levels=None, train_dataset=None, architecture_desc=None, device=None, **kwargs):
        if (architecture_desc is None) and (train_dataset is None):
            raise ValueError("train_dataset cannot = None if architecture_desc=None")
        super().__init__()
        self.problem_type = problem_type
        if self.problem_type == QUANTILE:
            self.register_buffer("quantile_levels", torch.Tensor(quantile_levels).float().reshape(1, -1))
        self.device = torch.device("cpu") if device is None else device
        if architecture_desc is None:
            params = self._set_params(**kwargs)
            # adpatively specify network architecture based on training dataset
            self.from_logits = False
            self.has_vector_features = train_dataset.has_vector_features
            self.has_embed_features = train_dataset.has_embed_features
            if self.has_embed_features:
                num_categs_per_feature = train_dataset.getNumCategoriesEmbeddings()
                embed_dims = get_embed_sizes(train_dataset, params, num_categs_per_feature)
            if self.has_vector_features:
                vector_dims = train_dataset.data_list[train_dataset.vectordata_index].shape[-1]
        else:
            # ignore train_dataset, params, etc. Recreate architecture based on description:
            self.architecture_desc = architecture_desc
            self.has_vector_features = architecture_desc["has_vector_features"]
            self.has_embed_features = architecture_desc["has_embed_features"]
            self.from_logits = architecture_desc["from_logits"]
            params = architecture_desc["params"]
            if self.has_embed_features:
                num_categs_per_feature = architecture_desc["num_categs_per_feature"]
                embed_dims = architecture_desc["embed_dims"]
            if self.has_vector_features:
                vector_dims = architecture_desc["vector_dims"]
        # init input size
        input_size = 0

        # define embedding layer:
        if self.has_embed_features:
            self.embed_blocks = nn.ModuleList()
            for i in range(len(num_categs_per_feature)):
                self.embed_blocks.append(nn.Embedding(num_embeddings=num_categs_per_feature[i], embedding_dim=embed_dims[i]))
                input_size += embed_dims[i]

        # update input size
        if self.has_vector_features:
            input_size += vector_dims

        # activation
        act_fn = nn.Identity()
        if params["activation"] == "elu":
            act_fn = nn.ELU()
        elif params["activation"] == "relu":
            act_fn = nn.ReLU()
        elif params["activation"] == "tanh":
            act_fn = nn.Tanh()

        layers = []
        if params["use_batchnorm"]:
            layers.append(nn.BatchNorm1d(input_size))
        layers.append(nn.Linear(input_size, params["hidden_size"]))
        layers.append(act_fn)
        for _ in range(params["num_layers"] - 1):
            if params["use_batchnorm"]:
                layers.append(nn.BatchNorm1d(params["hidden_size"]))
            layers.append(nn.Dropout(params["dropout_prob"]))
            layers.append(nn.Linear(params["hidden_size"], params["hidden_size"]))
            layers.append(act_fn)
        layers.append(nn.Linear(params["hidden_size"], num_net_outputs))
        self.main_block = nn.Sequential(*layers)

        if self.problem_type in [REGRESSION, QUANTILE]:  # set range for output
            y_range = params["y_range"]  # Used specifically for regression. = None for classification.
            self.y_constraint = None  # determines if Y-predictions should be constrained
            if y_range is not None:
                if y_range[0] == -np.inf and y_range[1] == np.inf:
                    self.y_constraint = None  # do not worry about Y-range in this case
                elif y_range[0] >= 0 and y_range[1] == np.inf:
                    self.y_constraint = "nonnegative"
                elif y_range[0] == -np.inf and y_range[1] <= 0:
                    self.y_constraint = "nonpositive"
                else:
                    self.y_constraint = "bounded"
                self.y_lower = y_range[0]
                self.y_upper = y_range[1]
                self.y_span = self.y_upper - self.y_lower

        if self.problem_type == QUANTILE:
            self.alpha = params["alpha"]  # for huber loss
        if self.problem_type == SOFTCLASS:
            self.log_softmax = torch.nn.LogSoftmax(dim=1)
        if self.problem_type in [BINARY, MULTICLASS, SOFTCLASS]:
            self.softmax = torch.nn.Softmax(dim=1)
        if architecture_desc is None:  # Save Architecture description
            self.architecture_desc = {
                "has_vector_features": self.has_vector_features,
                "has_embed_features": self.has_embed_features,
                "params": params,
                "num_net_outputs": num_net_outputs,
                "from_logits": self.from_logits,
            }
            if self.has_embed_features:
                self.architecture_desc["num_categs_per_feature"] = num_categs_per_feature
                self.architecture_desc["embed_dims"] = embed_dims
            if self.has_vector_features:
                self.architecture_desc["vector_dims"] = vector_dims

    def _set_params(
        self,
        num_layers=4,
        hidden_size=128,
        activation="relu",
        use_batchnorm=False,
        dropout_prob=0.1,
        y_range=None,
        alpha=0.01,
        max_embedding_dim=100,
        embed_exponent=0.56,
        embedding_size_factor=1.0,
    ):
        return dict(
            num_layers=num_layers,
            hidden_size=hidden_size,
            activation=activation,
            use_batchnorm=use_batchnorm,
            dropout_prob=dropout_prob,
            y_range=y_range,
            alpha=alpha,
            max_embedding_dim=max_embedding_dim,
            embed_exponent=embed_exponent,
            embedding_size_factor=embedding_size_factor,
        )

    def init_params(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, data_batch):
        input_data = []
        input_offset = 0
        if self.has_vector_features:
            input_data.append(data_batch[0].to(self.device))
            input_offset += 1
        if self.has_embed_features:
            embed_data = data_batch[input_offset]
            for i in range(len(self.embed_blocks)):
                input_data.append(self.embed_blocks[i](embed_data[i].to(self.device)))

        if len(input_data) > 1:
            input_data = torch.cat(input_data, dim=1)
        else:
            input_data = input_data[0]

        output_data = self.main_block(input_data)
        if self.problem_type in [REGRESSION, QUANTILE]:  # output with y-range
            if self.y_constraint is None:
                return output_data
            else:
                if self.y_constraint == "nonnegative":
                    return self.y_lower + torch.abs(output_data)
                elif self.y_constraint == "nonpositive":
                    return self.y_upper - torch.abs(output_data)
                else:
                    return torch.sigmoid(output_data) * self.y_span + self.y_lower
        elif self.problem_type == SOFTCLASS:
            return self.log_softmax(output_data)  # KLDivLoss takes in log-probabilities as predictions.
        else:
            return output_data

    def huber_pinball_loss(self, input_data, target_data):
        error_data = target_data.contiguous().reshape(-1, 1) - input_data
        if self.alpha == 0.0:
            loss_data = torch.max(self.quantile_levels * error_data, (self.quantile_levels - 1) * error_data)
            return loss_data.mean()

        loss_data = torch.where(torch.abs(error_data) < self.alpha, 0.5 * error_data * error_data, self.alpha * (torch.abs(error_data) - 0.5 * self.alpha))
        loss_data /= self.alpha
        scale = torch.where(error_data >= 0, torch.ones_like(error_data) * self.quantile_levels, torch.ones_like(error_data) * (1 - self.quantile_levels))
        loss_data *= scale
        return loss_data.mean()

    def margin_loss(self, input_data, margin_scale=0.0001):
        # number of samples
        batch_size, num_quantiles = input_data.size()

        # compute margin loss (batch_size x num_net_outputs(above) x num_net_outputs(below))
        error_data = input_data.unsqueeze(1) - input_data.unsqueeze(2)

        # margin data (num_quantiles x num_quantiles)
        margin_data = self.quantile_levels.permute(1, 0) - self.quantile_levels
        margin_data = torch.tril(margin_data, -1) * margin_scale

        # compute accumulated margin
        loss_data = torch.tril(error_data + margin_data, diagonal=-1)
        loss_data = loss_data.relu()
        loss_data = loss_data.sum() / float(batch_size * (num_quantiles * num_quantiles - num_quantiles) * 0.5)
        return loss_data

    def quantile_loss(self, predict_data, target_data, margin):
        if margin > 0.0:
            m_loss = self.margin_loss(predict_data)
        else:
            m_loss = 0.0
        h_loss = self.huber_pinball_loss(predict_data, target_data).mean()
        return h_loss + margin * m_loss

    def compute_loss(self, data_batch, loss_function=None, gamma=None):
        # train mode
        self.train()
        predict_data = self(data_batch)
        target_data = data_batch[-1].to(self.device)
        if self.problem_type in [BINARY, MULTICLASS]:
            target_data = target_data.type(torch.long)  # Windows default int type is int32. Need to explicit convert to Long.
        if self.problem_type == QUANTILE:
            return self.quantile_loss(predict_data, target_data, margin=gamma)
        if self.problem_type == SOFTCLASS:
            return loss_function(predict_data, target_data)
        else:
            target_data = target_data.flatten()
            if self.problem_type == REGRESSION:
                predict_data = predict_data.flatten()
            return loss_function(predict_data, target_data)

    def predict(self, input_data):
        self.eval()
        with torch.no_grad():
            predict_data = self(input_data)
            if self.problem_type == QUANTILE:
                predict_data = torch.sort(predict_data, -1)[0]  # sorting ensures monotonicity of quantile estimates
            elif self.problem_type in [BINARY, MULTICLASS, SOFTCLASS]:
                predict_data = self.softmax(predict_data)  # convert NN output to probability
            elif self.problem_type == REGRESSION:
                predict_data = predict_data.flatten()
            if self.problem_type == BINARY:
                predict_data = predict_data[:, 1]
            return predict_data.data.cpu().numpy()
