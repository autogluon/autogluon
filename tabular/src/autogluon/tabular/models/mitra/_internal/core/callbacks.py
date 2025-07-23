import numpy as np
import torch


class EarlyStopping():

    def __init__(self, patience=10, delta=0.0001, metric='log_loss'):

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.metric = metric


    def __call__(self, val_loss):
        
        # smaller is better for these metrics
        if self.metric in ["log_loss", "mse", "mae", "rmse"]:
            score = -val_loss
        # larger is better for these metrics
        elif self.metric in ["accuracy", "roc_auc", "r2"]:
            score = val_loss
        else:
            raise ValueError(f"Unsupported metric: {self.metric}. Supported metrics are: log_loss, mse, mae, rmse, accuracy, roc_auc, r2.")

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


class Checkpoint():

    def __init__(self):
        self.curr_best_loss = np.inf
        self.best_model: dict
    
    def reset(self, net: torch.nn.Module):
        self.curr_best_loss = np.inf
        self.best_model = net.state_dict()
        for key in self.best_model:
            self.best_model[key] = self.best_model[key].to('cpu')
        

    def __call__(self, net: torch.nn.Module, loss: float):
        
        if loss < self.curr_best_loss:
            self.curr_best_loss = loss
            self.best_model = net.state_dict()
            for key in self.best_model:
                self.best_model[key] = self.best_model[key].to('cpu')


    def set_to_best(self, net):
        net.load_state_dict(self.best_model)


class EpochStatistics():

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
    
class TrackOutput():

    def __init__(self) -> None:
        self.y_true: list[np.ndarray] = []
        self.y_pred: list[np.ndarray] = []

    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def get(self):
        return np.concatenate(self.y_true, axis=0), np.concatenate(self.y_pred, axis=0)