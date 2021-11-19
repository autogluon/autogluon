import numpy as np
import torch

class TemperatureScaling():
    def __init__(self, lr:float=):
        self.weights_init = weights_init
        self.logit_input = logit_input
        self.logit_constant = logit_constant
        self.reg_lambda_list = reg_lambda_list
        self.reg_mu_list = reg_mu_list
        self.initializer = initializer
        self.ref_row = ref_row