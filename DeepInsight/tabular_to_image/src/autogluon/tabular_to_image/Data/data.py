import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torchvision.transforms import *
from torch.utils.data import DataLoader
import torch
import numpy as np
from collections import namedtuple
import pandas as pd
import time
import os
import copy
from autogluon.core.dataset import TabularDataset

class ImageGerator:
    Dataset = TabularDataset
    def __init__(self):  
        
    @staticmethod
    def __get_dataset(data):
        if isinstance(data, TabularDataset):
            return data
        elif isinstance(data, pd.DataFrame):
            return TabularDataset(data)
        elif isinstance(data, str):
            return TabularDataset(data)
        elif isinstance(data, pd.Series):
            raise TypeError("data must be TabularDataset or pandas.DataFrame, not pandas.Series. \
                   To predict on just single example (ith row of table), use data.iloc[[i]] rather than data.iloc[i]")
        else:
            raise TypeError("data must be TabularDataset or pandas.DataFrame or str file path to data")
       
    
    