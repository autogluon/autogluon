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