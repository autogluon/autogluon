"Implements various metrics to measure training accuracy"

import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression


def isotonic(input_data, quantile_list):
    quantile_list = np.array(quantile_list).reshape(-1)
    batch_size = input_data.shape[0]
    new_output_data = []
    for i in range(batch_size):
        new_output_data.append(IsotonicRegression().fit_transform(quantile_list, input_data[i]))
    return np.stack(new_output_data, 0)


class HuberPinballLoss(nn.Module):
    __name__ = "huber_pinball_loss"

    def __init__(self, quantile_levels, alpha=0.01):
        super(HuberPinballLoss, self).__init__()
        if quantile_levels is not None:
            self.quantile_levels = torch.Tensor(quantile_levels).contiguous().reshape(1, -1)
        else:
            self.quantile_levels = None
        self.alpha = alpha

    def forward(self, predict_data, target_data):
        if self.quantile_levels is None:
            return None
        target_data = target_data.contiguous().reshape(-1, 1)
        batch_size = target_data.size()[0]
        predict_data = predict_data.contiguous().reshape(batch_size, -1)

        error_data = target_data - predict_data
        if self.alpha == 0.0:
            loss_data = torch.max(self.quantile_levels * error_data, (self.quantile_levels - 1) * error_data)
        else:
            loss_data = torch.where(torch.abs(error_data) < self.alpha, 0.5 * error_data * error_data, self.alpha * (torch.abs(error_data) - 0.5 * self.alpha))
            loss_data = loss_data / self.alpha

            scale = torch.where(error_data >= 0, torch.ones_like(error_data) * self.quantile_levels, torch.ones_like(error_data) * (1 - self.quantile_levels))
            loss_data *= scale
        return loss_data.mean()
