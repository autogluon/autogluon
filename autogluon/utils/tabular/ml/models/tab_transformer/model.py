
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.TabTransformer import TabTransformer, TabTransformer_modified

import code.readouts as readouts

class Model(nn.Module):
    def __init__(self,
                 **kwargs):
        
        super().__init__()
        self.head_dim           = kwargs['head_dim']
        self.kwargs             = kwargs
        self.readout            = readouts.__dict__[kwargs['readout']]

        self.embed  = TabTransformer(**kwargs['tab_kwargs'], **kwargs)
        self.head   = nn.Linear(self.embed.hidden_dim, self.head_dim)
        
    def forward(self, data):
        feature = self.embed(data)
        head  = F.normalize(self.head(feature), dim=-1)

        return feature, head #self.readout(*tuple([feature])), self.readout(*tuple([head]))

class Model_modified(nn.Module):
    def __init__(self,
                 **kwargs): 
        super().__init__()
        self.head_dim           = kwargs['head_dim']
        self.kwargs             = kwargs
        self.readout            = readouts.__dict__[kwargs['readout']]

        self.embed  = TabTransformer_modified(**kwargs['tab_kwargs'], **kwargs)
        self.head   = nn.Linear(self.embed.hidden_dim, self.head_dim)

        
    def forward(self, data):
        feature = self.embed(data)
        head  = F.normalize(self.head(feature), dim=-1)
        return feature, head #self.readout(*tuple([feature])), self.readout(*tuple([head]))





