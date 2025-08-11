"""
These are not callbacks, but if you look for callback most likely you look for these
"""

import io
from pathlib import Path

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0001):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def we_should_stop(self):
        return self.early_stop


class Checkpoint:
    def __init__(self, dirname: Path = None, id: str = None, in_memory: bool = True, save_best: bool = True):
        self.dirname = dirname
        self.id = id
        self.curr_best_loss = np.inf
        self.save_best = save_best
        self.in_memory = in_memory
        if not self.in_memory:
            assert self.dirname is not None
            assert self.id is not None
            self.path = Path(self.dirname) / f"params_{self.id}.pt"
            self.path.parent.mkdir(exist_ok=True)
        else:
            self.path = None

        self.buffer = io.BytesIO()
        self.best_model = None
        self.best_epoch = None

    def reset(self):
        self.curr_best_loss = np.inf
        self.best_model = None
        self.best_epoch = None
        self.buffer = io.BytesIO()

    def __call__(self, net, loss: float, epoch: int):
        if loss < self.curr_best_loss:
            self.curr_best_loss = loss
            self.best_model = net.state_dict()
            self.best_epoch = epoch
            if self.save_best:
                self.save()

    def save(self):
        if self.in_memory:
            self.buffer = io.BytesIO()
            torch.save(self.best_model, self.buffer)  # nosec B614
        else:
            torch.save(self.best_model, self.path)  # nosec B614

    def load(self):
        if self.in_memory:
            # Reset the buffer's position to the beginning
            self.buffer.seek(0)
            return torch.load(self.buffer, weights_only=True)  # nosec B614
        else:
            return torch.load(self.path)  # nosec B614


class EpochStatistics:
    def __init__(self) -> None:
        self.n = 0
        self.loss = 0
        self.score = 0

    def update(self, loss, score, n):
        self.n += n
        self.loss += loss * n
        self.score += score * n

    def get(self):
        return self.loss / self.n, self.score / self.n


class TrackOutput:
    def __init__(self) -> None:
        self.y_true: list[np.ndarray] = []
        self.y_pred: list[np.ndarray] = []

    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def get(self):
        return np.concatenate(self.y_true, axis=0), np.concatenate(self.y_pred, axis=0)
