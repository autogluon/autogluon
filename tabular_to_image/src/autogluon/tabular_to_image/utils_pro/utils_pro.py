import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch
#device = torch.device("cuda") #device = 'cuda'
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from autogluon.core.dataset import TabularDataset
from autogluon.DeepInsight.pyDeepInsight import ImageTransformer
class Utils_pro:
    def __init__(self, X_train_img ,X_val_img,X_test_img,y_train,y_val,y_test):
      self.X_train_img=X_train_img
      self.X_val_img=X_val_img
      self.X_test_img=X_test_img
      self.y_train=y_train
      self.y_val=y_val
      self.y_test=y_test
    


    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
    
    def _validate_fit_data(self, train_data, tuning_data=None, unlabeled_data=None):
        if isinstance(train_data, str):
            train_data = TabularDataset(train_data)
        if tuning_data is not None and isinstance(tuning_data, str):
            tuning_data = TabularDataset(tuning_data)
        if unlabeled_data is not None and isinstance(unlabeled_data, str):
            unlabeled_data = TabularDataset(unlabeled_data)

        if not isinstance(train_data, pd.DataFrame):
            raise AssertionError(f'train_data is required to be a pandas DataFrame, but was instead: {type(train_data)}')

        if len(set(train_data.columns)) < len(train_data.columns):
            raise ValueError("Column names are not unique, please change duplicated column names (in pandas: train_data.rename(columns={'current_name':'new_name'})")
        if tuning_data is not None:
            if not isinstance(tuning_data, pd.DataFrame):
                raise AssertionError(f'tuning_data is required to be a pandas DataFrame, but was instead: {type(tuning_data)}')
            train_features = [column for column in train_data.columns if column != self.label]
            tuning_features = [column for column in tuning_data.columns if column != self.label]
            if self.sample_weight is not None:
                if self.sample_weight in train_features:
                    train_features.remove(self.sample_weight)
                if self.sample_weight in tuning_features:
                    tuning_features.remove(self.sample_weight)
            if self._learner.groups is not None:
                train_features.remove(self._learner.groups)
            train_features = np.array(train_features)
            tuning_features = np.array(tuning_features)
            if np.any(train_features != tuning_features):
                raise ValueError("Column names must match between training and tuning data")
        if unlabeled_data is not None:
            if not isinstance(unlabeled_data, pd.DataFrame):
                raise AssertionError(f'unlabeled_data is required to be a pandas DataFrame, but was instead: {type(unlabeled_data)}')
            train_features = [column for column in train_data.columns if column != self.label]
            unlabeled_features = [column for column in unlabeled_data.columns]
            if self.sample_weight is not None:
                if self.sample_weight in train_features:
                    train_features.remove(self.sample_weight)
                if self.sample_weight in unlabeled_features:
                    unlabeled_features.remove(self.sample_weight)
            train_features = sorted(np.array(train_features))
            unlabeled_features = sorted(np.array(unlabeled_features))
            if np.any(train_features != unlabeled_features):
                raise ValueError("Column names must match between training and unlabeled data.\n"
                                 "Unlabeled data must have not the label column specified in it.\n")
        return train_data, tuning_data, unlabeled_data    
    def image_tensor(self): 
        preprocess = transforms.Compose([transforms.ToTensor()])    
        batch_size = 64
        
        le = LabelEncoder()
        #num_classes = np.unique(le.fit_transform(self.y_train)).size
        X_train_tensor = torch.stack([preprocess(img) for img in self.X_train_img])
        y_train_tensor = torch.from_numpy(le.fit_transform(self.y_train))

        X_val_tensor = torch.stack([preprocess(img) for img in self.X_val_img])
        y_val_tensor = torch.from_numpy(le.fit_transform(self.y_val ))

        X_test_tensor = torch.stack([preprocess(img) for img in self.X_test_img])
        y_test_tensor = torch.from_numpy(le.transform(self.y_test))
        
        trainset = TensorDataset(X_train_tensor, y_train_tensor)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        valset = TensorDataset(X_val_tensor, y_val_tensor)
        valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

        Testset = TensorDataset(X_test_tensor, y_test_tensor)
        Testloader = DataLoader(Testset, batch_size=batch_size, shuffle=True)
        return trainloader,valloader,Testloader#,num_classes 