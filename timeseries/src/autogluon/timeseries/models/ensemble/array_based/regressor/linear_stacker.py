from typing import Literal, Optional

import numpy as np
from typing_extensions import Self

from autogluon.timeseries.utils.timer import Timer

from .abstract import EnsembleRegressor


class LinearStackerEnsembleRegressor(EnsembleRegressor):
    """Linear stacker ensemble regressor using PyTorch optimization with softmax weights.

    Implements weighted averaging of base model predictions with learnable weights optimized
    via gradient descent. Uses PyTorch during training for optimization, then stores weights
    as numpy arrays for efficient prediction.

    Parameters
    ----------
    quantile_levels
        List of quantile levels for quantile predictions (e.g., [0.1, 0.5, 0.9]).
    weights_per
        Weight configuration specifying which dimensions to learn weights for:
        - "m": Per-model weights (shape: num_models), defaults to "m"
        - "mt": Per-model and per-time weights (shape: prediction_length, num_models)
        - "mq": Per-model and per-model-output (quantiles and mean) weights
          (shape: num_quantiles+1, num_models)
        - "mtq": Per-model, per-time, and per-quantile weights
          (shape: prediction_length, num_quantiles+1, num_models)
    lr
        Learning rate for Adam optimizer. Defaults to 0.1.
    max_epochs
        Maximum number of training epochs. Defaults to 10000.
    relative_tolerance
        Convergence tolerance for relative loss change between epochs. Defaults to 1e-7.
    """

    def __init__(
        self,
        quantile_levels: list[float],
        weights_per: Literal["m", "mt", "mq", "mtq"] = "m",
        lr: float = 0.1,
        max_epochs: int = 10_000,
        relative_tolerance: float = 1e-7,
    ):
        super().__init__()
        self.quantile_levels = quantile_levels
        self.weights_per = weights_per
        self.lr = lr
        self.max_epochs = max_epochs
        self.relative_tolerance = relative_tolerance

        # Learned weights (stored as numpy arrays)
        self.weights: Optional[np.ndarray] = None

    def _compute_weight_shape(self, base_model_predictions_shape: tuple) -> tuple:
        """Compute weight tensor shape based on weights_per configuration."""
        _, _, prediction_length, num_outputs, num_models = base_model_predictions_shape

        shapes = {
            "m": (1, 1, num_models),
            "mt": (prediction_length, 1, num_models),
            "mq": (1, num_outputs, num_models),
            "mtq": (prediction_length, num_outputs, num_models),
        }
        try:
            return (1, 1) + shapes[self.weights_per]
        except KeyError:
            raise ValueError(f"Unsupported weights_per: {self.weights_per}")

    def make_weighted_average_module(self, base_model_predictions_shape: tuple):
        import torch

        class WeightedAverage(torch.nn.Module):
            def __init__(self, shape):
                super().__init__()
                self.raw_weights = torch.nn.Parameter(torch.zeros(*shape, dtype=torch.float32))

            def get_normalized_weights(self):
                return torch.softmax(self.raw_weights, dim=-1)  # softmax over models

            def forward(self, base_model_predictions: torch.Tensor):
                return torch.sum(self.get_normalized_weights() * base_model_predictions, dim=-1)

        return WeightedAverage(self._compute_weight_shape(base_model_predictions_shape))

    def fit(
        self,
        base_model_mean_predictions: np.ndarray,
        base_model_quantile_predictions: np.ndarray,
        labels: np.ndarray,
        time_limit: Optional[float] = None,
    ) -> Self:
        import torch

        def _ql(
            labels_tensor: torch.Tensor,
            ensemble_predictions: torch.Tensor,
        ) -> torch.Tensor:
            """Compute the weighted quantile loss on predictions and ground truth (labels).
            Considering that the first dimension of predictions is the mean, we treat
            mean predictions on the same footing as median (0.5) predictions as contribution
            to the overall weighted quantile loss.
            """
            quantile_levels = torch.tensor([0.5] + self.quantile_levels, dtype=torch.float32)
            error = labels_tensor - ensemble_predictions  # (num_windows, num_items, num_time, num_outputs)
            quantile_loss = torch.maximum(quantile_levels * error, (quantile_levels - 1) * error)
            return torch.mean(quantile_loss)

        timer = Timer(time_limit).start()

        base_model_predictions = torch.tensor(
            np.concatenate(
                [base_model_mean_predictions, base_model_quantile_predictions],
                axis=3,
            ),
            dtype=torch.float32,
        )
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        weighted_average = self.make_weighted_average_module(base_model_predictions.shape)

        optimizer = torch.optim.Adam(weighted_average.parameters(), lr=self.lr)

        prev_loss = float("inf")
        for _ in range(self.max_epochs):
            optimizer.zero_grad()

            ensemble_predictions = weighted_average(base_model_predictions)

            loss = _ql(labels_tensor, ensemble_predictions)
            loss.backward()
            optimizer.step()

            loss_change = abs(prev_loss - loss.item()) / (loss.item() + 1e-8)
            if loss_change < self.relative_tolerance:
                break
            prev_loss = loss.item()

            if timer.timed_out():
                break

        # store final weights as numpy array
        # TODO: add sparsification to ensure negligible weights are dropped
        with torch.no_grad():
            self.weights = weighted_average.get_normalized_weights().detach().numpy()

        return self

    def predict(
        self,
        base_model_mean_predictions: np.ndarray,
        base_model_quantile_predictions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.weights is None:
            raise ValueError("Model must be fitted before prediction")

        # combine base model predictions
        all_predictions = np.concatenate([base_model_mean_predictions, base_model_quantile_predictions], axis=3)

        # predict
        ensemble_pred = np.sum(self.weights * all_predictions, axis=-1)

        mean_predictions = ensemble_pred[:, :, :, :1]
        quantile_predictions = ensemble_pred[:, :, :, 1:]

        return mean_predictions, quantile_predictions
