import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
#device = torch.device("cuda") #device = 'cuda'
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import numpy as np
from autogluon.core.dataset import TabularDataset
from pyDeepInsight import ImageTransformer,LogScaler 
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as tic
import autogluon.tabular_to_image.models_zoo 
from autogluon.tabular_to_image.models_zoo.models_zoo import ModelsZoo
from sklearn.manifold import TSNE
class Utils_pro:
    def __init__(self ,**kwargs):
        #self.train_dataset=train_dataset
        #self.labels=labels
             
              
        ModelsZoo_type = kwargs.pop('ModelsZoo_type', ModelsZoo)
        ModelsZoo_kwargs = kwargs.pop('ModelsZoo_kwargs', dict())
        ImageShape = kwargs.get('ImageShape', None)
        self._ModelsZoo_type = type(self._ModelsZoo)
        self._ModelsZoo: ModelsZoo = ModelsZoo_type(ImageShape=ImageShape ,**ModelsZoo_kwargs)
        #model_type = kwargs.get('model_type', None)
        #num_classes = kwargs.get('num_classes', None)
        #pretrained = kwargs.get('pretrained', None)
      
    Dataset = TabularDataset  
    models_count=self._ModelsZoo.len_group_counts() 
    groups_counts=self._ModelsZoo.new_countsD()
    g1_presentage=round(((groups_counts['g1']/models_count)*100),1)/100
    g2_presentage=round(((groups_counts['g2']/models_count)*100),1)/100
    g3_presentage=round(((groups_counts['g3']/models_count)*100),1)/100
    g4_presentage=round(((groups_counts['g4']/models_count)*100),1)/100
    g5_presentage=round(((groups_counts['g5']/models_count)*100),1)/100
    #def data_split(self,):
    #    X_train, X_test, y_train, y_test = train_test_split(self.train_dataset,  self.label_column, test_size=0.2)
    #    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25
   
   
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
    
    
    @property
    def len_group_counts(self):
        return self._ModelsZoo.ModelsZoo.len_group_counts 
    
    @property
    def new_countsD(self):
        return self._ModelsZoo.ModelsZoo.new_countsD 
    
    @property
    def ImageShape(self):
        return self._ModelsZoo.ImageShape 
    
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
             
    def spit_dataset(self,data):
        models_count=self.len_group_counts() 
        groups_counts=self.new_countsD()
        data = self.__get_dataset(data)
       
        if isinstance(data, str):
            data = TabularDataset(data)
        if not isinstance(data, pd.DataFrame):
            raise AssertionError(f'data is required to be a pandas DataFrame, but was instead: {type(data)}')
        if len(set(data.columns)) < len(data.columns):
                raise ValueError("Column names are not unique, please change duplicated column names (in pandas: train_data.rename(columns={'current_name':'new_name'})")                           
        
        data_g1=data.sample(frac=self.g1_presentage, replace=False, random_state=12)
        data_g2=data.sample(frac=self.g2_presentage, replace=False, random_state=44)
        data_g3=data.sample(frac=self.g3_presentage, replace=False, random_state=58)
        data_g4=data.sample(frac=self.g4_presentage, replace=False, random_state=11)
        data_g5=data.sample(frac=self.g5_presentage, replace=False, random_state=77)
        
        return data_g1,data_g2,data_g3,data_g4,data_g5
        
      
    def _validate_fit_data(self, data,labels):        
        data_g1,data_g2,data_g3,data_g4,data_g5=self.spit_dataset(data)
        X1 = data_g1.drop(columns=[labels], axis=1) #data_g1.iloc[:,:-1]  #independent columns df6_pd.drop('PERMIT_TYPE', axis=1)
        Y1=  data_g1[labels].values#data_g1.iloc[:,-1]
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,Y1, test_size=0.2)
        X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train1, y_train1, test_size=0.25)
        if X_val1 is not None:
            if not isinstance(X_val1, pd.DataFrame):
                raise AssertionError(f'X_val1 is required to be a pandas DataFrame, but was instead: {type(X_val1)}')
            train_features1 = [column for column in X_train1.columns if column !=labels]
            val_features1 = [column for column in X_val1.columns if column != labels]
            if np.any(train_features1 != val_features1):
                raise ValueError("Column names must match between training and val data")
            if X_test1 is not None:
                if not isinstance(X_test1 , pd.DataFrame):
                    raise AssertionError(f'X_test1 is required to be a pandas DataFrame, but was instead: {type(X_test1)}')
            train_features1 = [column for column in X_train1.columns if column !=labels]
            test_features1 = [column for column in X_test1.columns]
            if np.any(train_features1 != test_features1):
                raise ValueError("Column names must match between training and test_data")
        Dic_data_g1={'X_train1':X_train1,'X_test1':X_test1,'y_train1':y_train1,'y_test1':y_test1,'X_val1':X_val1,' y_val1': y_val1}    
        
        X2 = data_g2.drop(columns=[labels], axis=1) #data_g1.iloc[:,:-1]  #independent columns df6_pd.drop('PERMIT_TYPE', axis=1)
        Y2=  data_g2[labels].values#data_g1.iloc[:,-1]
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,Y2, test_size=0.2)
        X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train2, y_train2, test_size=0.25)
        if X_val2 is not None:
            if not isinstance(X_val2, pd.DataFrame):
                raise AssertionError(f'X_val2 is required to be a pandas DataFrame, but was instead: {type(X_val2)}')
            train_features2 = [column for column in X_train2.columns if column !=labels]
            val_features2 = [column for column in X_val2.columns if column != labels]
            if np.any(train_features2 != val_features2):
                raise ValueError("Column names must match between training and val data")
            if X_test2 is not None:
                if not isinstance(X_test2 , pd.DataFrame):
                    raise AssertionError(f'X_test2 is required to be a pandas DataFrame, but was instead: {type(X_test2)}')
            train_features2 = [column for column in X_train2.columns if column !=labels]
            test_features2 = [column for column in X_test2.columns]
            if np.any(train_features2 != test_features2):
                raise ValueError("Column names must match between training and test_data")
        Dic_data_g2={'X_train2':X_train2,'X_test2':X_test2,'y_train2':y_train2,'y_test2':y_test2,'X_val2':X_val2,' y_val2': y_val2}  
        
        X3 = data_g3.drop(columns=[labels], axis=1) #data_g1.iloc[:,:-1]  #independent columns df6_pd.drop('PERMIT_TYPE', axis=1)
        Y3=  data_g3[labels].values#data_g1.iloc[:,-1]
        X_train3, X_test3, y_train3, y_test3 = train_test_split(X3,Y3, test_size=0.2)
        X_train3, X_val3, y_train3, y_val3 = train_test_split(X_train3, y_train3, test_size=0.25)
        if X_val3 is not None:
            if not isinstance(X_val3, pd.DataFrame):
                raise AssertionError(f'X_val3 is required to be a pandas DataFrame, but was instead: {type(X_val3)}')
            train_features3 = [column for column in X_train3.columns if column !=labels]
            val_features3 = [column for column in X_val3.columns if column != labels]
            if np.any(train_features3 != val_features3):
                raise ValueError("Column names must match between training and val data")
            if X_test3 is not None:
                if not isinstance(X_test3 , pd.DataFrame):
                    raise AssertionError(f'X_test3 is required to be a pandas DataFrame, but was instead: {type(X_test3)}')
            train_features3 = [column for column in X_train3.columns if column !=labels]
            test_features3 = [column for column in X_test3.columns]
            if np.any(train_features3 != test_features3):
                raise ValueError("Column names must match between training and test_data")
        Dic_data_g3={'X_train3':X_train3,'X_test3':X_test3,'y_train3':y_train3,'y_test3':y_test3,'X_val3':X_val3,' y_val3': y_val3} 
        
        X4 = data_g4.drop(columns=[labels], axis=1) #data_g1.iloc[:,:-1]  #independent columns df6_pd.drop('PERMIT_TYPE', axis=1)
        Y4=  data_g4[labels].values#data_g1.iloc[:,-1]
        X_train4, X_test4, y_train4, y_test4 = train_test_split(X4,Y4, test_size=0.2)
        X_train4, X_val4, y_train4, y_val4 = train_test_split(X_train4, y_train4, test_size=0.25)
        if X_val4 is not None:
            if not isinstance(X_val4, pd.DataFrame):
                raise AssertionError(f'X_val4 is required to be a pandas DataFrame, but was instead: {type(X_val4)}')
            train_features4 = [column for column in X_train4.columns if column !=labels]
            val_features4 = [column for column in X_val4.columns if column != labels]
            if np.any(train_features4 != val_features4):
                raise ValueError("Column names must match between training and val data")
            if X_test3 is not None:
                if not isinstance(X_test4 , pd.DataFrame):
                    raise AssertionError(f'X_test4 is required to be a pandas DataFrame, but was instead: {type(X_test4)}')
            train_features4 = [column for column in X_train4.columns if column !=labels]
            test_features4 = [column for column in X_test4.columns]
            if np.any(train_features4 != test_features4):
                raise ValueError("Column names must match between training and test_data")
        Dic_data_g4={'X_train4':X_train4,'X_test4':X_test4,'y_train4':y_train4,'y_test4':y_test4,'X_val4':X_val4,' y_val4': y_val4}
        
        X5 = data_g5.drop(columns=[labels], axis=1) #data_g1.iloc[:,:-1]  #independent columns df6_pd.drop('PERMIT_TYPE', axis=1)
        Y5=  data_g5[labels].values#data_g1.iloc[:,-1]
        X_train5, X_test5, y_train5, y_test5 = train_test_split(X5,Y5, test_size=0.2)
        X_train5, X_val5, y_train5, y_val5 = train_test_split(X_train5, y_train5, test_size=0.25)
        if X_val4 is not None:
            if not isinstance(X_val5, pd.DataFrame):
                raise AssertionError(f'X_val4 is required to be a pandas DataFrame, but was instead: {type(X_val4)}')
            train_features5 = [column for column in X_train5.columns if column !=labels]
            val_features5 = [column for column in X_val5.columns if column != labels]
            if np.any(train_features5 != val_features5):
                raise ValueError("Column names must match between training and val data")
            if X_test3 is not None:
                if not isinstance(X_test5 , pd.DataFrame):
                    raise AssertionError(f'X_test4 is required to be a pandas DataFrame, but was instead: {type(X_test4)}')
            train_features5 = [column for column in X_train5.columns if column !=labels]
            test_features5 = [column for column in X_test5.columns]
            if np.any(train_features5 != test_features5):
                raise ValueError("Column names must match between training and test_data")
        Dic_data_g5={'X_train5':X_train5,'X_test5':X_test5,'y_train5':y_train5,'y_test5':y_test5,'X_val5':X_val5,' y_val5': y_val5}
        return Dic_data_g1,Dic_data_g2,Dic_data_g3,Dic_data_g4,Dic_data_g5
                
               
    def Image_Genartor(self,data):
        data=self.__get_dataset(data)
        ln = LogScaler()
        X_train,X_val,X_test,_ , _,_=self._validate_fit_data(data)
        X_train_norm = ln.fit_transform(X_train)
        X_val_norm = ln.fit_transform(X_val)
        X_test_norm = ln.transform(X_test)
        #@jit(target ="cuda") 
        model=self._ModelsZoo.ModelsZoo.create_model()
        it = ImageTransformer(feature_extractor='tsne',pixels=self.ImageShape, random_state=1701,n_jobs=-1)
       
        X_train_img = it.fit_transform(X_train_norm)
        X_val_img = it.fit_transform(X_val_norm)
        X_test_img = it.transform(X_test_norm)

        tsne = TSNE(n_components=2, perplexity=30, metric='cosine',random_state=1701, n_jobs=-1)

        plt.figure(figsize=(5, 5))
        _ = it.fit(X_train_norm, plot=True)
        return X_train_img,X_val_img,X_test_img
    
    def len_of_Images(self,data):
        X_train_img,X_val_img,X_test_img=self.Image_Genartor(data)
        return len(X_train_img),len(X_val_img),len(X_test_img)
        
    def image_tensor(self,data): 
        preprocess = transforms.Compose([transforms.ToTensor()])    
        batch_size = 64
        
        le = LabelEncoder()
        #num_classes = np.unique(le.fit_transform(self.y_train)).size
        data=self.__get_dataset(data)
        _,_,_,y_train , y_val,y_test=self._validate_fit_data(data)
        X_train_img,X_val_img,X_test_img=self.Image_Genartor(data)
        X_train_tensor = torch.stack([preprocess(img) for img in X_train_img ])
        y_train_tensor = torch.from_numpy(le.fit_transform(y_train))

        X_val_tensor = torch.stack([preprocess(img) for img in X_val_img])
        y_val_tensor = torch.from_numpy(le.fit_transform(y_val ))

        X_test_tensor = torch.stack([preprocess(img) for img in X_test_img])
        y_test_tensor = torch.from_numpy(le.transform(y_test))
        
        trainset = TensorDataset(X_train_tensor, y_train_tensor)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        valset = TensorDataset(X_val_tensor, y_val_tensor)
        valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

        Testset = TensorDataset(X_test_tensor, y_test_tensor)
        Testloader = DataLoader(Testset, batch_size=batch_size, shuffle=True)
        return trainloader,valloader,Testloader#,num_classes 
    
    
#python3 -m pip install  pydot
#python3 -m pip install pydotplus
#sudo apt-get install graphviz libgraphviz-dev pkg-config
#sudo apt-get install python-pip python-virtualenv
#python3 -m pip install  pygraphviz
#python3 -m pip install -e DeepInsight/
#python3 -m  pip -q install git+git://github.com/alok-ai-lab/DeepInsight.git#egg=DeepInsight    
#from autogluon.tabular_to_image.utils_pro import Utils_pro