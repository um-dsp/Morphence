# -*- coding: utf-8 -*-
"""

@author: Abderrahmen Amich
@email:  aamich@umich.edu
"""


import os
import pickle
import numpy as np
import time
import torch
import torchvision
from tqdm import tqdm
from easydict import EasyDict
from random import random
from cifar_data import get_datasets, CNN
from train_mnist import PyNet, ld_mnist

# Get current working directory
cwd = os.getcwd()

# Create execution device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True


def load_model(dir_path,ind,data,adv_train=False):
    '''load model'''
    
    if data == 'MNIST':
        model = PyNet()
    elif data == 'CIFAR10':
        model = CNN()

    model.cuda()
    if adv_train==False:
        model.load_state_dict(torch.load(os.path.join(cwd,dir_path,"CNN_"+data + str(ind) + ".pth")))
    else:
        model.load_state_dict(torch.load(os.path.join(cwd,dir_path,"R-CNN_"+data + str(ind) + ".pth")))
    
    return model

class Morphence():
    
    '''Morphence prediction system'''
    
    def __init__(self, test_size, Q_max,n,data,starting_batch, class_nb):
        
        self.test_size = test_size
        self.Q_max = Q_max
        self.n = n
        self.data = data
        #self.lamda = lamda
        self.starting_batch = starting_batch
        self.class_nb = class_nb
        
        self.nb_queries = 0 # total number of queries previously performed on Morphence
        self.scheduling={} # number of selections for each model
        for i in range(self.n+1): 
            self.scheduling[i]=0
        
        # distribution of test set over different pool of models
        self.queries = list(range(0,self.test_size+1,self.Q_max)) # queries limits for each pool of models
        if self.test_size % self.Q_max != 0:
            self.queries.append(self.test_size % self.Q_max + self.queries[-1])
        
        print('The distribution of {} queries with respect to Q_max = {} is {}.'.format(self.test_size,self.Q_max,self.queries))
        
    def predict2(self,x):
        ''' predict the labels of a set x using highest conf scheduling of MTD'''
        
        #print(self.nb_queries)
        #print('Received {} queries'.format(x.shape[0]))
        
        y_probs=[] # prediction probabilities  of all models
        
        for qi in range(len(self.queries)-1):
            
            if self.nb_queries >= self.queries[qi] and self.nb_queries < self.queries[qi+1]:
                if self.nb_queries + x.shape[0] <= self.queries[qi+1]:
                    models=[]
                    for i in range(1,self.n+1):
                        try:
                            #models.append(load_model(os.path.join(cwd,'experiments',self.data,self.data+"_models_"+''.join(str(self.lamda).split('.'))+'_'+str(self.n)+'_'+self.starting_batch[0]+str(int(self.starting_batch[1])+qi)),i))
                            models.append(load_model(os.path.join(cwd,'experiments',self.data,self.data+"_models_"+self.starting_batch[0]+str(int(self.starting_batch[1])+qi)),i,self.data))
                        except FileNotFoundError:
                            raise('model {} is not found'.format(i))
                            
                    #print('### Responding to queries from {} to {} using models pool {}'.format(self.nb_queries+1,self.nb_queries + x.shape[0],self.starting_batch[0]+str(int(self.starting_batch[1])+qi)))
                    for model in models:
                        if device == "cuda":
                            model = model.cuda()
                        model.eval()
                        y_probs.append(model(x))
                
                elif self.nb_queries + x.shape[0] > self.queries[qi+1]:
                    models1=[]
                    models2=[]
                    x1 = x[:self.queries[qi+1]-self.nb_queries]
                    #print('### Responding to queries from {} to {} using models pool {}'.format(self.nb_queries+1,self.queries[qi+1],self.starting_batch[0]+str(int(self.starting_batch[1])+qi)))
                    if self.queries[qi+1] < self.test_size:
                        x2 = x[self.queries[qi+1]-self.nb_queries:]
                        #print('And responding to queries from {} to {} using models pool {}'.format(self.queries[qi+1]+1,self.nb_queries + x.shape[0],self.starting_batch[0]+str(int(self.starting_batch[1])+qi+1)))
                    for i in range(1,self.n+1):
                        try:
                            #models1.append(load_model(os.path.join(cwd,'experiments',self.data,self.data+"_models_"+''.join(str(self.lamda).split('.'))+'_'+str(self.n)+'_'+self.starting_batch[0]+str(int(self.starting_batch[1])+qi)),i))
                            models1.append(load_model(os.path.join(cwd,'experiments',self.data,self.data+"_models_"+self.starting_batch[0]+str(int(self.starting_batch[1])+qi)),i,self.data))
                        except FileNotFoundError:
                            raise('model {} is not found'.format(i))
                            
                        if self.queries[qi+1] < self.test_size:
                            try:
                                #models2.append(load_model(os.path.join(cwd,'experiments',self.data,self.data+"_models_"+''.join(str(self.lamda).split('.'))+'_'+str(self.n)+'_'+self.starting_batch[0]+str(int(self.starting_batch[1])+qi+1)),i))
                                models2.append(load_model(os.path.join(cwd,'experiments',self.data,self.data+"_models_"+self.starting_batch[0]+str(int(self.starting_batch[1])+qi+1)),i,self.data))
                            except FileNotFoundError:
                                raise('model {} is not found'.format(i))
                                
                    if self.queries[qi+1] < self.test_size:
                        for model1,model2 in zip(models1,models2):
                            if device == "cuda":
                                model1 = model1.cuda()
                                model2 = model2.cuda()
                            model1.eval()
                            model2.eval()
                            
                            y_probs.append(torch.cat((model1(x1),model2(x2)),dim=0))
                    else:
                        
                        for model1 in models1:
                            if device == "cuda":
                                model1 = model1.cuda()
                                
                            model1.eval()
                            y_probs.append(model1(x))
                            
        
            
        
        # update number of queries                       
        self.nb_queries += x.shape[0]
        
        # Select the model that has the highest prediction confidence
        for ind in range(x.shape[0]):
            for i in range(self.n):
                
                if i==0:
                    #print(y_probs[i].shape)
                    y_ind = torch.reshape(y_probs[i][ind], (1, self.class_nb))
                else:
                    y_ind = torch.cat((y_ind,torch.reshape(y_probs[i][ind], (1, self.class_nb))),dim=0)
                
            #print('predictions of sample {} are {}'.format(ind,y_ind))
            #print('highest confidence vector of sample {} are {}'.format(ind, y_ind.max(0)[0]))
            selected_ind = y_ind.max(0)[1][y_ind.max(0)[0].argmax().item()].item()
            #print('selected model for sample {} is model {}'.format(ind, selected_ind))
            
            # keep track of selected models
            self.scheduling[selected_ind]+=1
            
            if ind == 0:
                y_pred = torch.reshape(y_ind.max(0)[0], (1, self.class_nb))
            else:
                y_pred = torch.cat((y_pred,torch.reshape(y_ind.max(0)[0], (1, self.class_nb))))
                
            
        
        
        return y_pred
