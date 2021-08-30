import matplotlib.pyplot as plt
import time
import os
import copy
import torch
#device = torch.device("cuda") #device = 'cuda'
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from autogluon.tabular_to_image.utils_pro import  Utils_pro
from autogluon.tabular_to_image.models_zoo import ModelsZoo
from autogluon.tabular_to_image.models_zoo import ModelsZoo
class ImagePredictions:
    
    def init(self,**kwargs):
        self._validate_init_kwargs(kwargs)
        Utils_pro_type = kwargs.pop('Utils_pro_type', Utils_pro)
        Utils_pro_kwargs = kwargs.pop('Utils_pro_kwargs', dict())
              
       
        X_train_img = kwargs.get('X_train_img', None)
        X_val_img = kwargs.get('X_val_img', None)
        X_test_img = kwargs.get('X_test_img', None)
        
        y_train = kwargs.get('y_train', None)
        y_val = kwargs.get('y_val', None)
        y_test = kwargs.get('y_test', None)
        
        self._Utils_pro: Utils_pro = Utils_pro_type(X_train_img=X_train_img ,X_val_img=X_val_img,X_test_img=X_test_img,
                                        y_train=y_train,y_val=y_val,y_test=y_test,**Utils_pro_kwargs)
        self._Utils_pro_type = type(self._Utils_pro)
        
        
        
        ModelsZoo_type = kwargs.pop('ModelsZoo_type', ModelsZoo)
        ModelsZoo_kwargs = kwargs.pop('ModelsZoo_kwargs', dict())
        
                     
        ImageShape = kwargs.get('ImageShape', None)
        model_type = kwargs.get('model_type', None)
        num_classes = kwargs.get('num_classes', None)
        pretrained = kwargs.get('pretrained', None)
              
        self._ModelsZoo: ModelsZoo = ModelsZoo_type(ImageShape=ImageShape ,model_type=model_type,
                                        num_classes=num_classes,pretrained=pretrained,**Utils_pro_kwargs)
        self._ModelsZoo_type = type(self._ModelsZoo)
        #rainloader,valloader,Testloader =self._Utils_pro.Utils_pro.image_tensor()
        #criterion,optimizer,exp_lr_scheduler=self._ModelsZoo.ModelsZoo.optimizer()
        #use_gpu = torch.cuda.is_available()
        #models=self._ModelsZoo.ModelsZoo.create_model()
        
    
     
        
    @property
    def X_train_img(self):
        return self._Utils_pro.X_train_img 
    @property
    def X_val_img(self):
        return self._Utils_pro.X_val_img  
    @property
    def X_test_img(self):
        return self._Utils_pro.X_test_img    
    
    @property
    def y_train(self):
        return self._Utils_pro.y_train 
    @property
    def y_val(self):
        return self._Utils_pro.y_val 
    @property
    def y_test(self):
        return self._Utils_pro.y_test 
    
    @property
    def ImageShape(self):
        return self._ModelsZoo.ImageShape 
    @property
    def model_type(self):
        return self._ModelsZoo.model_type 
    @property
    def num_classes(self):
        return self._ModelsZoo.num_classes 
    
    @property
    def pretrained(self):
        return self._ModelsZoo.pretrained 
   
    @property
    def model(self):
        return self._ModelsZoo.ModelsZoo.create_model() 
     
    @staticmethod
    def _validate_init_kwargs(kwargs):
        valid_kwargs = {
            'Utils_pro_type',
            'Utils_pro_kwargs',
             'X_train_img',
             'X_val_img',
             'X_test_img',
             'y_train',
             'y_val,y_test',
             'ImageShape',
             'model_type',
             'num_classes',
             'pretrained',
             
       
        }
        invalid_keys = []
        for key in kwargs:
            if key not in valid_kwargs:
                invalid_keys.append(key)
        if invalid_keys:
            raise ValueError(f'Invalid kwargs passed: {invalid_keys}\nValid kwargs: {list(valid_kwargs)}')
    
    """
    def train(self,dataloader, model, num_epochs=20):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Rprop(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)

        model.train(True)
        results = []
        for epoch in range(num_epochs):
            optimizer.step()
            scheduler.step()
            model.train()

            running_loss = 0.0
            running_corrects = 0

            n = 0
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                n += len(labels)

            epoch_loss = running_loss / float(n)
            epoch_acc = running_corrects.double() / float(n)

            print(f'epoch {epoch}/{num_epochs} : {epoch_loss:.5f}, {epoch_acc:.5f}')
            results.append(EpochProgress(epoch, epoch_loss, epoch_acc.item()))
        return pd.DataFrame(results)
    """
  
    def train_model(self,model, num_epochs=3):
        #criterion = nn.CrossEntropyLoss() #optimizer = optim.Rprop(model.parameters(), lr=0.01) #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
        trainloader,valloader,_ =self._Utils_pro.Utils_pro.image_tensor()
        criterion,optimizer,_=self._ModelsZoo.ModelsZoo.optimizer()
        model=self._ModelsZoo.ModelsZoo.create_model()
        use_gpu = torch.cuda.is_available()
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        avg_loss = 0
        avg_acc = 0
        avg_loss_val = 0
        avg_acc_val = 0
        
        
        train_batches = len(trainloader)
        val_batches = len(valloader)
        
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs))
            print('-' * 10)
            
            loss_train = 0
            loss_val = 0
            acc_train = 0
            acc_val = 0
            
            model.train(True)
            
            for i, data in enumerate(trainloader):
                if i % 100 == 0:
                    print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)
                    
                # Use half training dataset
                #if i >= train_batches / 2:
                #    break
                    
                inputs, labels = data
                
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                #loss_train += loss.data[0]
                loss_train += loss.item() * inputs.size(0)
                acc_train += torch.sum(preds == labels.data)
                
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
            
            print()
            # * 2 as we only used half of the dataset
            avg_loss = loss_train * 2 /len(self.X_train_img) #dataset_sizes[TRAIN]
            avg_acc = acc_train * 2 /len(self.X_train_img)#dataset_sizes[TRAIN]
            
            model.train(False)
            model.eval()
                
            for i, data in enumerate(valloader):
                if i % 100 == 0:
                    print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
                    
                inputs, labels = data
                
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
                else:
                    inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                
                #loss_val += loss.data[0]
                loss_val += loss.item() * inputs.size(0)
                acc_val += torch.sum(preds == labels.data)
                
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
            
            avg_loss_val = loss_val /len(self.X_test_img) #dataset_sizes[VAL]
            avg_acc_val = acc_val /len(self.X_val_img) #dataset_sizes[VAL]
            
            print()
            print("Epoch {} result: ".format(epoch))
            print("Avg loss (train): {:.4f}".format(avg_loss))
            print("Avg acc (train): {:.4f}".format(avg_acc))
            print("Avg loss (val): {:.4f}".format(avg_loss_val))
            print("Avg acc (val): {:.4f}".format(avg_acc_val))
            print('-' * 10)
            print()
            
            if avg_acc_val > best_acc:
                    best_acc = avg_acc_val
                    best_model_wts = copy.deepcopy(model.state_dict())
                
            elapsed_time = time.time() - since
            print()
            print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
            print("Best acc: {:.4f}".format(best_acc))
            
            model.load_state_dict(best_model_wts)
            return model
    
    def eval_model(self):
        _,_,Testloader =self._Utils_pro.Utils_pro.image_tensor()
        criterion,_,_=self._ModelsZoo.ModelsZoo.optimizer()
        model=self._ModelsZoo.ModelsZoo.create_model()
        use_gpu = torch.cuda.is_available()
        since = time.time()
        avg_loss = 0
        avg_acc = 0
        loss_test = 0
        acc_test = 0
        
        test_batches = len(Testloader)
        print("Evaluating model")
        print('-' * 10)
        
        for i, data in enumerate(Testloader):
            if i % 100 == 0:
                print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

            model.train(False)
            model.eval()
            inputs, labels = data

            if use_gpu:
                inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
            else:
                inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            #loss_test += loss.data[0]
            loss_test += loss.item() * inputs.size(0)
            acc_test += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
            
        avg_loss = loss_test /len(self.X_test_img) #dataset_sizes[TEST]
        avg_acc = acc_test /len(self.X_test_img)#dataset_sizes[TEST]
        
        elapsed_time = time.time() - since
        print()
        print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
        print("Avg loss (test): {:.4f}".format(avg_loss))
        print("Avg acc (test): {:.4f}".format(avg_acc))
        print('-' * 10)
        
    """
    def plot_results(df, figsize=(10, 5)):
        fig, ax1 = plt.subplots(figsize=figsize)

        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss', color='tab:red')
        ax1.plot(df['epoch'], df['loss'], color='tab:red')

        ax2 = ax1.twinx()
        ax2.set_ylabel('accuracy', color='tab:blue')
        ax2.plot(df['epoch'], df['accuracy'], color='tab:blue')

        fig.tight_layout()
    """