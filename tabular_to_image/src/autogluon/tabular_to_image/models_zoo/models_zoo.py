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
#from autogluon.TablarToImage import  Utils
from efficientnet_pytorch import EfficientNet

class ModelsZoo():  
    def __init__(self, ImageShape,model_type, num_classes, pretrained=True):  
        self.ImageShape = ImageShape 
        self.model_type=model_type
        self.num_classes=num_classes
        self.pretrained=True
        #use_gpu = torch.cuda.is_available() 
         
    group_counts = {"g1": ['resnet18','resnet34','resnet50','resnet101','resnet152','alexnet','vgg11',
                            'vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19','vgg19_bn',
                            'densenet121','densenet161','densenet169','densenet201''googlenet',
                            'shufflenet_v2_x0_5','shufflenet_v2_x1_0','mobilenet_v2','wide_resnet50_2',
                            'wide_resnet101_2','mnasnet0_5','mnasnet1_0','efficientnet-b1',
                            'efficientnet-b2','efficientnet-b3','efficientnet-b4','efficientnet-b5',
                            'efficientnet-b6','efficientnet-b7'],"g2": ['squeezenet1_0','squeezenet1_1'],
                        "g3": ['resnext50_32x4d','resnext101_32x8d'],"g4": ['inception_v3']}
    len_group_counts=(sum([len(group_counts[x]) for x in group_counts if isinstance(group_counts[x], list)]))
    new_countsD =new_countsD = {k: len(v) for k,v in group_counts.items()}
    def create_model(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
      
        if   self.ImageShape=='224':
            if 'resnet18' == self.model_type:
                model = models.resnet18(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Linear(model.fc.in_features,self.num_classes ).double()
            elif 'resnet34' == self.model_type:
                model = models.resnet34(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Linear(model.fc.in_features, self.num_classes).double()  
            elif 'resnet50' == self.model_type:
                model = models.resnet50(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Linear(model.fc.in_features, self.num_classes).double()                
            elif 'resnet101' == self.model_type:
                model = models.resnet101(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Linear(model.fc.in_features, self.num_classes).double() 
            elif 'resnet152' == self.model_type:
                model = models.resnet152(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Linear(model.fc.in_features, self.num_classes).double()
            elif 'alexnet' == self.model_type:
                model = models.alexnet(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.classifier[6] = nn.Linear(4096, self.num_classes).double() 
            elif 'vgg11' == self.model_type:
                model = models.vgg11(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes).double()
            elif 'vgg11_bn' == self.model_type:
                model = models.vgg11_bn(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes).double()
            elif 'vgg13' == self.model_type:
                model = models.vgg13(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False    
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes).double()
            elif 'vgg13_bn' == self.model_type:
                model = models.vgg13_bn(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes).double()
            elif 'vgg16' == self.model_type:
                model = models.vgg16(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes).double()
            elif 'vgg16_bn' == self.model_type:
                model = models.vgg16_bn(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes).double()
            elif 'vgg19' == self.model_type:
                model = models.vgg19(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes).double()
            elif 'vgg19_bn' == self.model_type:
                model = models.vgg19_bn(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes).double()
            elif 'densenet121' == self.model_type:
                model = models.densenet121(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier = nn.Linear(model.classifier.in_features, self.num_classes).double()
            elif 'densenet161' == self.model_type:
                model = models.densenet161(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier = nn.Linear(model.classifier.in_features, self.num_classes).double()
            elif 'densenet169' == self.model_type:
                model = models.densenet169(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier = nn.Linear(model.classifier.in_features, self.num_classes).double()
            elif 'densenet201' == self.model_type:
                model = models.densenet201(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier = nn.Linear(model.classifier.in_features, self.num_classes).double()
            elif 'googlenet' == self.model_type:
                model = models.googlenet(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.fc = nn.Linear(model.fc.in_features, self.num_classes).double()
            elif 'shufflenet_v2_x0_5' == self.model_type:
                model = models.shufflenet_v2_x0_5(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.fc = nn.Linear(model.fc.in_features, self.num_classes).double()
            elif 'shufflenet_v2_x1_0' == self.model_type:
                model = models.shufflenet_v2_x1_0(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.fc = nn.Linear(model.fc.in_features, self.num_classes).double()
            elif 'mobilenet_v2' == self.model_type:
                model = models.mobilenet_v2(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes).double()
            elif 'wide_resnet50_2' == self.model_type:
                model = models.wide_resnet50_2(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.fc = nn.Linear(model.fc.in_features, self.num_classes).double()
            elif 'wide_resnet101_2' == self.model_type:
                model = models.wide_resnet101_2(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Linear(model.fc.in_features, self.num_classes).double
            elif 'mnasnet0_5' == self.model_type:
                model = models.mnasnet0_5(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
            elif 'mnasnet1_0' == self.model_type:
                model = models.mnasnet1_0(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes).double()         
            elif 'efficientnet-b1'==self.model_type:
                model = EfficientNet.from_name('efficientnet-b1')
                for param in model.parameters():
                   param.requires_grad = True  
                model._fc = nn.Linear(model._fc.in_features, self.num_classes).double() 
            elif 'efficientnet-b2'==self.model_type:
                model = EfficientNet.from_name('efficientnet-b2')
                for param in model.parameters():
                   param.requires_grad = True  
                model._fc = nn.Linear(model._fc.in_features, self.num_classes).double()
            elif 'efficientnet-b3'==self.model_type:
                model = EfficientNet.from_name('efficientnet-b3')
                for param in model.parameters():
                   param.requires_grad = True  
                model._fc = nn.Linear(model._fc.in_features, self.num_classes).double() 
            elif 'efficientnet-b4'==self.model_type:
                model = EfficientNet.from_name('efficientnet-b4')
                for param in model.parameters():
                   param.requires_grad = True  
                model._fc = nn.Linear(model._fc.in_features, self.num_classes).double() 
            elif 'efficientnet-b5'==self.model_type:
                model = EfficientNet.from_name('efficientnet-b5')
                for param in model.parameters():
                   param.requires_grad = True  
                model._fc = nn.Linear(model._fc.in_features, self.num_classes).double()
            elif 'efficientnet-b6'==self.model_type:
                model = EfficientNet.from_name('efficientnet-b6')
                for param in model.parameters():
                   param.requires_grad = True  
                model._fc = nn.Linear(model._fc.in_features, self.num_classes).double() 
            elif 'efficientnet-b7'==self.model_type:
                model = EfficientNet.from_name('efficientnet-b7')
                for param in model.parameters():
                   param.requires_grad = True  
                model._fc = nn.Linear(model._fc.in_features, self.num_classes).double()                 
        elif self.ImageShape=='227':
            if 'squeezenet1_0' == self.model_type:
                model = models.squeezenet1_0(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                    model.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1,1), stride=(1,1)).double()
                    model.num_classes = self.num_classes
            elif 'squeezenet1_1' == self.model_type:
                model = models.squeezenet1_1(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1,1), stride=(1,1)).double()
                model.num_classes = self.num_classes
        elif self.ImageShape=='256':
            if 'resnext50_32x4d' == self.model_type:
                model = models.resnext50_32x4d(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes).double()
            elif 'resnext101_32x8d' == self.model_type:
                model = models.resnext101_32x8d(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.fc = nn.Linear(model.fc.in_features, self.num_classes).double()      
        elif self.ImageShape=='299':
            if 'inception_v3' == self.model_type:
                model = models.inception_v3(pretrained=self.pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, self.num_classes).double()
                model.fc = nn.Linear(model.fc.in_features, self.num_classes).double()
        return model.double().to(device)
    
    def optimizer(self):
        criterion = nn.CrossEntropyLoss() 
        if self.model_type in []:
            #optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
            optimizer = optim.SGD(self.create_model().parameters(), lr=0.001, momentum=0.9)
            exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        elif self.model_type in []:
            optimizer=torch.optim.RMSprop(self.create_model(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
            # Decay LR by a factor of 0.1 every 7 epochs
            exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        elif self.model_type in []:
            optimizer = optim.Adam(self.create_model().parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
            exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience =  5, mode = 'max', verbose=True)       
        return   criterion,optimizer,exp_lr_scheduler

#np.random.seed(37)
#torch.manual_seed(37)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

#num_classes = 3
#pretrained = True
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#EpochProgress = namedtuple('EpochProgress', 'epoch, loss, accuracy')

#transform = transforms.Compose([Resize(224), ToTensor()])
#image_folder = datasets.ImageFolder('./shapes/train', transform=transform)
#dataloader = DataLoader(image_folder, batch_size=4, shuffle=True, num_workers=4)
    

