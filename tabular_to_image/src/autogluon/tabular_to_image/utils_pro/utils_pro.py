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
from DeepInsight.pyDeepInsight import ImageTransformer,LogScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
#from sklearn.model_selection import StratifiedKFol
from sklearn.manifold import TSNE
class Utils_pro:
    def __init__(self, label_column ):
      #self.train_dataset=train_dataset
      self.label_column=label_column
      
      
    Dataset = TabularDataset  
    #def data_split(self,):
    #    X_train, X_test, y_train, y_test = train_test_split(self.train_dataset,  self.label_column, test_size=0.2)
    #    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25
   
   
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
    
    
    def _validate_data(self, data):        
        data = self.__get_dataset(data)
        if isinstance(data, str):
            data = TabularDataset(data)
        if not isinstance(data, pd.DataFrame):
            raise AssertionError(f'data is required to be a pandas DataFrame, but was instead: {type(data)}')
        if len(set(data.columns)) < len(data.columns):
            raise ValueError("Column names are not unique, please change duplicated column names (in pandas: train_data.rename(columns={'current_name':'new_name'})")
        
        X_train, X_test, y_train, y_test = train_test_split(data,  self.label_column, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)
        if X_val is not None:
            if not isinstance(X_val, pd.DataFrame):
                raise AssertionError(f'X_val is required to be a pandas DataFrame, but was instead: {type(X_val)}')
            train_features = [column for column in X_train.columns if column != self.label_column]
            val_features = [column for column in X_val.columns if column != self.label_columnabel]
            if np.any(train_features != val_features):
                raise ValueError("Column names must match between training and val data")
        if X_test is not None:
            if not isinstance(X_test, pd.DataFrame):
                raise AssertionError(f'X_test is required to be a pandas DataFrame, but was instead: {type(X_test)}')
            train_features = [column for column in X_train.columns if column != self.label_columnabel]
            test_features = [column for column in X_test.columns]
            if np.any(train_features != test_features):
                raise ValueError("Column names must match between training and test_data")
         
        return X_train,X_val,X_test,y_train , y_val,y_test    
     
    
    
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
        
        
    def Image_Genartor(self,image_shape):
        ln = LogScaler()
        X_train,X_val,X_test,y_train , y_val,y_test=self._validate_data(data)
        X_train_norm = ln.fit_transform(X_train)
        X_val_norm = ln.fit_transform(X_val)
        X_test_norm = ln.transform(X_test)
        #@jit(target ="cuda") 
        it = ImageTransformer(feature_extractor='tsne',pixels=image_shape, random_state=1701,n_jobs=-1)
       
        X_train_img = it.fit_transform(X_train_norm)
        X_val_img = it.fit_transform(X_val_norm)
        X_test_img = it.transform(X_test_norm)

        tsne = TSNE(n_components=2, perplexity=30, metric='cosine',random_state=1701, n_jobs=-1)

        plt.figure(figsize=(5, 5))
        _ = it.fit(X_train_norm, plot=True)
        return X_train_img,X_val_img,X_test_img
        
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