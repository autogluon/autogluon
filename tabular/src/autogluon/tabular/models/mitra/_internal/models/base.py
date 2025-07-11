import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    
    def __init__(self):
        super().__init__()
    
    def init_weights(self):
        """Initialize model weights."""
        pass
    
    @abstractmethod
    def forward(self, 
        x_support: torch.Tensor, 
        y_support: torch.Tensor, 
        x_query: torch.Tensor, 
        **kwargs):
        """Forward pass for the model."""
        pass