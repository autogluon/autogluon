import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from typing_extensions import Self

from .abstract import EnsembleRegressor

logger = logging.getLogger(__name__)


class LinearStackerEnsembleRegressor(EnsembleRegressor):
    """Linear stacker ensemble regressor using PyTorch optimization with softmax weights."""

    def __init__(
        self,
        quantile_levels: list[float],
        weights_per: str = "m",
        lr: float = 0.1,
        max_epochs: int = 10000,
        tolerance_change: float = 1e-7,
        tolerance_grad: float = 1e-6,
    ):
        super().__init__()
        self.quantile_levels = quantile_levels
        self.weights_per = weights_per
        self.lr = lr
        self.max_epochs = max_epochs
        self.tolerance_change = tolerance_change
        self.tolerance_grad = tolerance_grad

        # Learned weights (stored as numpy arrays)
        self.weights: Optional[np.ndarray] = None
        self.weight_shape: Optional[tuple] = None

    def _compute_weight_shape(self, base_model_predictions_shape: tuple) -> tuple:
        """Compute weight tensor shape based on weights_per configuration."""
        num_windows, num_items, prediction_length, _, num_models = base_model_predictions_shape

        shape = []
        for char in self.weights_per:
            if char == "t":
                shape.append(prediction_length)
            elif char == "q":
                # Use the number of quantile levels + 1 for mean
                shape.append(len(self.quantile_levels) + 1)

        # Always add models dimension last
        shape.append(num_models)

        return tuple(shape)

    def _create_weight_module(self, weight_shape: tuple) -> nn.Module:
        """Create a simple PyTorch module for weight optimization."""

        class WeightModule(nn.Module):
            def __init__(self, shape):
                super().__init__()
                self.raw_weights = nn.Parameter(torch.zeros(*shape, dtype=torch.float64))

            def forward(self):
                # Apply softmax along the last dimension (models)
                return torch.softmax(self.raw_weights, dim=-1)

        return WeightModule(weight_shape)

    def _apply_weights(self, weights: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """Apply weights to predictions based on weights_per configuration."""
        if self.weights_per == "m":
            # weights: (num_models,), predictions: (windows, items, time, quantiles, models)
            return torch.sum(weights * predictions, dim=-1)
        elif self.weights_per == "mt":
            # weights: (time, models), predictions: (windows, items, time, quantiles, models)
            weights = weights.unsqueeze(0).unsqueeze(0).unsqueeze(3)  # (1, 1, time, 1, models)
            return torch.sum(weights * predictions, dim=-1)
        elif self.weights_per == "mq":
            # weights: (quantiles, models), predictions: (windows, items, time, quantiles, models)
            weights = weights.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # (1, 1, 1, quantiles, models)
            return torch.sum(weights * predictions, dim=-1)
        elif self.weights_per == "mtq":
            # weights: (time, quantiles, models), predictions: (windows, items, time, quantiles, models)
            weights = weights.unsqueeze(0).unsqueeze(0)  # (1, 1, time, quantiles, models)
            return torch.sum(weights * predictions, dim=-1)
        else:
            raise ValueError(f"Unsupported weights_per: {self.weights_per}")

    def fit(
        self,
        base_model_mean_predictions: np.ndarray,
        base_model_quantile_predictions: np.ndarray,
        labels: np.ndarray,
        time_limit: Optional[float] = None,
        **kwargs,
    ) -> Self:
        # Combine mean and quantile predictions
        all_predictions = np.concatenate([base_model_mean_predictions, base_model_quantile_predictions], axis=3)

        # Convert to PyTorch tensors
        predictions_tensor = torch.tensor(all_predictions, dtype=torch.float64)
        labels_tensor = torch.tensor(labels.squeeze(-1), dtype=torch.float64)  # Remove last dim

        # Compute weight shape based on combined predictions shape
        self.weight_shape = self._compute_weight_shape(all_predictions.shape)
        weight_module = self._create_weight_module(self.weight_shape)

        # Setup optimizer
        optimizer = torch.optim.Adam(weight_module.parameters(), lr=self.lr)

        # Training loop
        last_loss = float("inf")
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()

            weights = weight_module()
            ensemble_pred = self._apply_weights(weights, predictions_tensor)

            # Compute quantile loss for all quantiles (including mean as 0.5 quantile)
            all_quantiles = [0.5] + self.quantile_levels
            loss = 0.0
            for i, q in enumerate(all_quantiles):
                pred_q = ensemble_pred[:, :, :, i]
                error = labels_tensor - pred_q
                loss += torch.mean(torch.maximum(q * error, (q - 1) * error))

            loss.backward()
            optimizer.step()

            # Check convergence
            loss_change = abs(last_loss - loss.item())
            max_grad = max(p.grad.abs().max().item() for p in weight_module.parameters() if p.grad is not None)

            if loss_change < self.tolerance_change and max_grad < self.tolerance_grad:
                logger.debug(f"Converged after {epoch + 1} epochs")
                break

            last_loss = loss.item()

        # Store final weights as numpy array
        with torch.no_grad():
            self.weights = weight_module().detach().numpy()

        return self

    def predict(
        self,
        base_model_mean_predictions: np.ndarray,
        base_model_quantile_predictions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.weights is None:
            raise ValueError("Model must be fitted before prediction")

        # Combine predictions
        all_predictions = np.concatenate([base_model_mean_predictions, base_model_quantile_predictions], axis=3)

        # Apply weights using numpy operations
        if self.weights_per == "m":
            ensemble_pred = np.sum(self.weights * all_predictions, axis=-1)
        elif self.weights_per == "mt":
            weights = self.weights[np.newaxis, np.newaxis, :, np.newaxis, :]
            ensemble_pred = np.sum(weights * all_predictions, axis=-1)
        elif self.weights_per == "mq":
            weights = self.weights[np.newaxis, np.newaxis, np.newaxis, :, :]
            ensemble_pred = np.sum(weights * all_predictions, axis=-1)
        elif self.weights_per == "mtq":
            weights = self.weights[np.newaxis, np.newaxis, :, :, :]
            ensemble_pred = np.sum(weights * all_predictions, axis=-1)
        else:
            raise ValueError(f"Unsupported weights_per: {self.weights_per}")

        # Split mean and quantile predictions
        mean_predictions = ensemble_pred[:, :, :, :1]
        quantile_predictions = ensemble_pred[:, :, :, 1:]

        return mean_predictions, quantile_predictions
