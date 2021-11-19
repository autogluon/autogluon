import numpy as np
import scipy
import torch


class TemperatureScaling():
    def __init__(self, lr: float = 0.01, init_val: float = 1, max_iter: int = 1000, gamma: float = 0.99):
        """


        Parameters:
        -----------
        init_val: float
            Initial value for temperature scalar term
        max_iter: int
            The maximum number of iterations to step in tuning
        lr: float
            The initial learning rate

        Return:
        float: The temperature scaling term
        """
        self.lr = lr
        self.init_val = init_val
        self.max_iter = max_iter
        self.gamma = gamma
        self.temperature = None

    def fit(self, y_val_probs, y_val_labels):
        """
        Tunes a temperature scalar term that divides the logits produced by autogluon model. Logits are generated
        by natural log the predicted probs from model then divides by a temperature scalar, which is tuned
        to minimize cross entropy on validation set.

        Parameters:
        -----------
        y_val_probs: numpy ndarray
            Predictive probabilities by model on validation set
        y_val: numpy ndarray
            The labels to the validation set
        """
        y_val_tensor = torch.tensor(y_val_labels)

        if self.temperature is None:
            self.temperature = torch.nn.Parameter(torch.ones(1).fill_(self.init_val))
        logits = torch.tensor(np.log(y_val_probs))
        nll_criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=self.lr, max_iter=self.max_iter)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)

        def temperature_scale_step():
            optimizer.zero_grad()
            temp = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
            new_logits = (logits / temp)
            loss = nll_criterion(new_logits, y_val_tensor)
            loss.backward()
            scheduler.step()
            return loss

        optimizer.step(temperature_scale_step)

    def predict_proba(self, pred_probs):
        logits = np.log(pred_probs)
        return scipy.special.softmax(logits / self.temperature)
