# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 12:14:58 2021

@author: Abderrahmen Amich
"""


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

def get_transform():
    '''
    @return: torch transforms to use in CIFAR10
    '''
    transform = transforms.Compose([
        transforms.Resize( (32,32) ),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))        
    ])
    
    return transform

def get_datasets(root='data', train=True, test=True, transform=None, batch=128, model='target'):
    '''
    @brief: function that obtains the CIFAR10 dataset and return
            the referent DataLoaders
    @param root: place to store original data
    @param train: when True, returns the train data
    @param test: when True, returns the test data
    @param batch: batch size to dataloaders
    @return: dictionary composed by: 'train' and 'test' datasets and 
                                      the name of the 'classes'.
             Dictionary keys: 'train', 'test', 'classes'
    '''
    assert train or test, 'You must select train, test, or both'
    ret = {}
    transform = get_transform() if transform is None else transform
    if train:
        trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform
        )
        if model=='copycat':
            shuffle=False
        else:
            shuffle=True
            
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch, shuffle=shuffle, num_workers=2
        )
        
        ret['train']   = trainloader
        ret['n_train'] = len(trainset)
        
        return EasyDict(train=trainloader)
        
    if test:
        testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch, shuffle=False, num_workers=2
        )
        ret['test']   = testloader
        ret['n_test'] = len(testset)
        
        return EasyDict(test=testloader)
    '''
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    ret['classes'] = classes
    '''
    



class CNN(nn.Module):
    """Sample model."""

    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)

        return x