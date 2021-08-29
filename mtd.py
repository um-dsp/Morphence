# -*- coding: utf-8 -*-
"""


@author: 
"""


import os
import pickle
import numpy as np
import time
import torch
import torchvision
from tqdm import tqdm
from train_mnist import PyNet, ld_mnist
from absl import app, flags
from easydict import EasyDict
from random import random
from cleverhans.torch.attacks.projected_gradient_descent import (projected_gradient_descent,)
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.spsa import spsa
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cifar import get_datasets, CNN
#from cifar_data import get_datasets, CNN

FLAGS = flags.FLAGS

# Get current working directory
cwd = os.getcwd()

# Create execution device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

supported_datasets = ['MNIST','CIFAR10']
supported_attacks = ['NoAttack','CW', 'FGS', 'SPSA']

    
def load_data(train=True,test=True):
    ''' load dataset'''
    
    if FLAGS.data=='MNIST':
        data = ld_mnist(batch_size=FLAGS.batch)
    elif FLAGS.data=='CIFAR10':
        data = get_datasets(root=os.path.join(cwd,'Copycat','Framework','data'),train=train, batch=FLAGS.batch)
    elif FLAGS.data not in supported_datasets:
        raise ValueError('Dataset {} is not supported'.format(FLAGS.data))

    return data


def load_model(dir_path,ind,adv_train=False,model_type='mtd',copycat=False):
    '''load model'''
    
    
    if FLAGS.data not in supported_datasets:
        raise ValueError('Dataset {} is not supported'.format(FLAGS.data))
    
    if FLAGS.data == 'MNIST':
        model = PyNet()
    elif FLAGS.data == 'CIFAR10':
        model = CNN()
    
    model.cuda()
    if adv_train==False:
        if copycat==False:
            model.load_state_dict(torch.load(os.path.join(cwd,dir_path,"CNN_"+FLAGS.data + str(ind) + ".pth")))
        else:
            model.load_state_dict(torch.load(os.path.join(cwd,dir_path,'copycat_'+model_type+'_'+"CNN_"+FLAGS.data + str(ind) + ".pth")))
    else:
        if copycat==False:
            model.load_state_dict(torch.load(os.path.join(cwd,dir_path,"R-CNN_"+FLAGS.data + str(ind) + ".pth")))
        else:
            model.load_state_dict(torch.load(os.path.join(cwd,dir_path,'copycat_'+model_type+'_'+"R-CNN_"+FLAGS.data + str(ind) + ".pth")))
    
    return model
        
    
        
def accuracy(model,data, size, model_type='torch'):
    '''compute accuracy'''
    
    if model_type=='torch':
        if device == "cuda":
            model = model.cuda()
        model.eval()
        
    shape=0
    report = EasyDict(nb_test=0, correct=0)
    for x, y in data.test:
        x, y = x.to(device), y.to(device)
        _, y_pred = model(x).max(1)
        report.nb_test += y.size(0)
        report.correct += y_pred.eq(y).sum().item()
        shape+=x.shape[0]
        if model_type=='mtd':
            print('Current accuracy is :{} %'.format(report.correct / report.nb_test * 100.0))
        if shape >= size:
            break
            
    return report.correct / report.nb_test * 100.0


def perform_attack(model,data,size,attack='spsa',model_type='mtd',copycat=False):
    '''perform a cleverhans attack
    
    model_type: can be either mtd or master or master_adv'''
    i=0
    correct=0
    nb_test=0
    for x, y in data.test:
        #print(x.shape)
        x, y = x.to(device), y.to(device)
        print('# Performing {} attack on batch {}'.format(attack,i+1))
        if attack=='spsa':
            x = spsa(model, x,eps=FLAGS.eps,nb_iter=10,norm = np.inf,sanity_checks=False)
        if attack=='CW':
            x = carlini_wagner_l2(model, x, 10,y,targeted = False)
            
        if attack=='FGS':
            x = fast_gradient_method(model, x, eps=FLAGS.eps, norm = np.inf)
        
        if model_type=='mtd':
            if attack != 'spsa':
                model.eval()
        _, y_pred = model(x).max(1)
        nb_test += x.shape[0]
        correct += y_pred.eq(y).sum().item()
        print('Current robustness against {} is :{} %'.format(attack,correct/nb_test *100))
        # save adv test data
        if i==0:
            x_adv = x
            y_adv = y
        else:
            x_adv = torch.cat((x_adv,x))
            y_adv = torch.cat((y_adv,y))
        i+=1
        print(x_adv.shape)
        print(y_adv.shape)
        if x_adv.shape[0]>=size:
            break
        
    if model_type=='master':
        if copycat==False:
            dir_path = os.path.join(cwd,attack+'_test_master')
        else:
            dir_path = os.path.join(cwd,attack+'copycat_test_master')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        f = open(os.path.join(cwd,attack+'_test_master',FLAGS.data), 'wb')
        pickle.dump((x_adv,y_adv), f)
        f.close()
    
    if model_type=='master_adv':
        if copycat==False:
            dir_path = os.path.join(cwd,attack+'_test_master_adv')
        else:
            dir_path = os.path.join(cwd,attack+'copycat_test_master_adv')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        f = open(os.path.join(cwd,attack+'_test_master_adv',FLAGS.data), 'wb')
        pickle.dump((x_adv,y_adv), f)
        f.close()
        
    if model_type=='mtd':
        if copycat == False:
            dir_path=os.path.join(cwd,attack+'_test')
        else:
            dir_path = os.path.join(cwd,attack+'_test_copycat')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        f = open(os.path.join(dir_path,FLAGS.data), 'wb')
        pickle.dump((x_adv,y_adv), f)
        f.close()
        
    #if model_type=='student':
    #    dir_path=os.path.join(cwd,attack+'_student_adv')
    #    if not os.path.exists(dir_path):
    #        os.makedirs(dir_path)
        
    #    f = open(os.path.join(dir_path,FLAGS.data), 'wb')
    #    pickle.dump((x_adv,y_adv), f)
    #    f.close()
    
    return x_adv, y_adv

def robustness(model,data,size,batch_size=128,attack='CW',model_type='student',copycat=False):
    '''Comupte the accuracy under attack'''
    
    print('model_type= ',model_type)
    print('Copycat = ', copycat)
    if model_type=='master_adv':
        if copycat==False:
            path = os.path.join(cwd,attack+'_test_master_adv',FLAGS.data)
        else:
            path = os.path.join(cwd,attack+'copycat_test_master_adv',FLAGS.data)
     
        
    elif model_type in ['master','student']:
        if copycat==False:
            path = os.path.join(cwd,attack+'_test_master',FLAGS.data)
        else:
            path = os.path.join(cwd,attack+'copycat_test_master',FLAGS.data)
        if attack in ['CW','FGS']:
            if copycat==False:
                path = os.path.join(cwd,attack+'_test',FLAGS.data)
            else:
                path = os.path.join(cwd,attack+'copycat_test_master',FLAGS.data)
                
            
    
    else:
        if attack=='spsa':
            dir_path=os.path.join(cwd,attack+'_test'+''.join(str(FLAGS.lamda).split('.'))+'_'+str(FLAGS.p))
            if model_type=='mtd' and FLAGS.p>4:
                dir_path=os.path.join(cwd,attack+'_test'+''.join(str(FLAGS.lamda).split('.'))+'_'+str(5))
            elif model_type=='mtd' and FLAGS.p<=4:
                dir_path=os.path.join(cwd,attack+'_test'+''.join(str(FLAGS.lamda).split('.'))+'_'+str(3))
            
        else:
            if copycat==False:
                dir_path=os.path.join(cwd,attack+'_test')
            else:
                dir_path=os.path.join(cwd,attack+'_test_copycat')
        path = os.path.join(dir_path,FLAGS.data)
    
    
    print(path)
    if not os.path.exists(path):
        if model_type in ['master_adv','master'] and copycat==False:
            if device == "cuda":
                model = model.cuda()
            model.eval()
        elif model_type in ['student','mtd']:
            if copycat==False:
                master_model = load_model(cwd,'')
                if device == "cuda":
                    master_model = master_model.cuda()
                master_model.eval()
            else:
                target_model=load_model(os.path.join(cwd,'Copycat','Framework'),'',model_type=model_type,copycat=True)
                if device == "cuda":
                    target_model = target_model.cuda()
                target_model.eval()
        elif model_type in ['master_adv','master'] and copycat==True:
            target_model=load_model(os.path.join(cwd,'Copycat','Framework'),'',model_type=model_type,copycat=True)
            if device == "cuda":
                target_model = target_model.cuda()
            target_model.eval()
            
        if attack in ['CW', 'FGS']:
            #if attack=='CW':
            #    size=1000
                    
            if model_type in ['student','master','mtd']:
                if copycat==False:
                    x_adv, y_adv = perform_attack(master_model,data,size,attack=attack,model_type='mtd')
                else:
                    x_adv, y_adv = perform_attack(target_model,data,size,attack=attack,model_type=model_type,copycat=True)
            
                
            if model_type in ['master_adv']:
                if copycat==False:
                    x_adv, y_adv = perform_attack(model,data,size,attack=attack,model_type=model_type)
                else:
                    x_adv, y_adv = perform_attack(target_model,data,size,attack=attack,model_type=model_type,copycat=copycat)
            
                    
        else:
            x_adv, y_adv = perform_attack(model,data,size,attack=attack,model_type=model_type)
            
    
    else:
        #print('Loading ', path)
        f = open(path, 'rb')
        x_adv,y_adv = pickle.load(f)
        f.close()
    
    if model_type in ['student', 'master', 'master_adv']:
        if device == "cuda":
            model = model.cuda()
        model.eval()
        
    report = EasyDict(nb_test=0, correct=0)
    shape=0
    for i in range(0,x_adv.shape[0],batch_size):
        x, y = get_batch(x_adv,y_adv, i, batch_size)
        x, y = x.to(device), y.to(device)
        _, y_pred = model(x).max(1)
        report.nb_test += y.size(0)
        report.correct += y_pred.eq(y).sum().item() 
        shape+=x.shape[0]
        if model_type=='mtd':
            print('Current robustness against {} is :{} %'.format(attack,report.correct / report.nb_test * 100.0))
        del x, y
        torch.cuda.empty_cache()
        
        if shape >= size:
            break
        
    return report.correct / report.nb_test * 100.0
        

def get_batch(x_all,y_all, start, batch_size=128):
    '''Get a batch of data'''
    
    y=[]
    i = start
    while i<start+batch_size and i<x_all.shape[0]:
        if i==start:
            x = x_all[i:i+1]
            y.append(y_all[i].item())
        elif i>start:
            x = torch.cat((x,x_all[i:i+1]))
            y.append(y_all[i].item())
        i+=1
        
        
    y=torch.LongTensor(y)
    
    
    return x, y
            
def retrain(net,data,device,epochs,batch_size=128,transform=None,adversarial=False,save=False):
    '''retrain network either on new clean data or adversarial data'''
    
    if device == "cuda":
        net = net.cuda()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    if FLAGS.data == 'MNIST':
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    elif FLAGS.data == 'CIFAR10':
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train model
    net.train()
    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        
        i=0
        start_time = time.time()
        with tqdm(data.train, unit="batch") as tepoch:
            for x, y in tepoch:
                x, y = x.to(device), y.to(device)
                #if i%2==0:
                if transform != None:
                    x=transform(x)
                        
                       
                if adversarial==True: 
                    
                    #if i%2==0:
                    # Replace clean example with adversarial example for adversarial training
                    master_model = load_model(cwd,'')
                    
                    if i%100==0 and FLAGS.data=='MNIST':
                        #print('performing CW on batch {} for adversarial training'.format(i)) 
                        x = carlini_wagner_l2(master_model, x, 10,y,targeted = False)
                    else: 
                        #print('performing PGD on batch {} for adversarial training'.format(i))
                        x = projected_gradient_descent(master_model, x, FLAGS.eps, 0.01, 40, np.inf)
                            
                            
                        
                
                optimizer.zero_grad()
                loss = loss_fn(net(x), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                    
                i+=1
                
        print("--- A  epoch takes %s seconds ---" % (time.time() - start_time))
        print("epoch: {}/{}, train loss: {:.3f}".format(
                epoch, epochs, train_loss))    
            
        
            
        
    acc = accuracy(net,data,FLAGS.test_set)
                                                                                           
    return net, acc


def perturb_weights_and_retrain(master_path,data,lamda,n,p,batch_size=128,new_train=False):
    
    '''Generate n diverse and accurate student models where
        p of the n models are adversarially trained
        
    :param master_path: path to the state_dict of master model
    :param data: test data loader
    :param lamda: the exponential decay of laplacian noise
    :param n: number of student models to generate
    :param p: number of adversarially trained student models
    '''
    
    for i in range(n):
        print('## Generating student model {} ##'.format(i+1))
        
        # load master model
        model = load_model(os.path.join(master_path),'')
        master_acc = accuracy(model,data,FLAGS.test_set)
        print("test acc of master model (%): {:.3f}".format(master_acc))
        rep=0
        while True:
            model = load_model(os.path.join(master_path),'')
            
        
            for param_tensor in model.state_dict():
                
                shape = model.state_dict()[param_tensor].size()
                
                # laplace mechanism
                if device == "cuda":
                    try:
                        model.state_dict()[param_tensor]+= torch.cuda.FloatTensor(np.random.laplace(loc=0.0, scale=lamda, size=shape))
                    except RuntimeError:
                        model.state_dict()[param_tensor]+= torch.cuda.LongTensor(np.random.laplace(loc=0.0, scale=lamda, size=shape))
                else:
                    model.state_dict()[param_tensor]+= np.random.laplace(loc=0.0, scale=lamda, size=shape)
            acc = accuracy(model,data,FLAGS.test_set)
            rep+=1
            if acc>10:
                break
            if rep==5:
                print('student model is not retrainable please try noise scale lower than {}'.format(FLAGS.lamda))
                #pass
                return
        print("Acc of student model {} after weight perturbation (%): {:.3f}".format(i+1,acc))
        
        # retrain student model
        trans_shift = 0.1+(random()*(0.2-0.1)) #scaled value = min + (value * (max - min))
        rot_deg = 10+(random()*(20-10))
        transform=torchvision.transforms.RandomAffine(degrees=rot_deg, translate=(trans_shift,trans_shift))
        
        
        epoch=0
        while True:
            if epoch%5==0:
                old_acc=acc
            epoch+=1
            if new_train==False:
                model, acc = retrain(model,data,device,1,batch_size=FLAGS.batch,transform=None,adversarial=False)
            else:
                model, acc = retrain(model,data,device,1,batch_size=FLAGS.batch,transform=transform,adversarial=False)
            if acc < old_acc:
                old_acc=acc
            print("Accuracy of student model after {} epochs of retraining (%): {:.3f}".format(epoch,acc))
            if epoch%5==0:
                print('Old_acc :',old_acc)
                if acc >= np.floor(master_acc) or acc-old_acc<0.5:
                    break
     
    
        print("Acc of student model {} after retraining (%): {:.3f}".format(i+1,acc))
        
        if i >= n-p:
            old_acc = acc
            start_time = time.time()
            it = 0
            #if n-i <= p:
            print('# Performing adversarial training on student model {}'.format(i+1))
            epoch=0
            old_rob = robustness(model,data,1000,batch_size,attack='FGS')
            max_rob = old_rob
            print("Robustness of student model {} before adversarial training (%) is {:.3f}".format(i+1,old_rob))
            while True:
                epoch+=1
                model, acc = retrain(model,data,device,1,batch_size=FLAGS.batch,transform=transform,adversarial=True)
                rob = robustness(model,data,1000,batch_size,attack='FGS')
                it +=1
                if rob > max_rob:
                    max_rob = rob
                if rob < old_rob:
                    old_rob=rob
                if epoch%7==0:
                    print('old_rob',old_rob)
                    if rob-old_rob<1:
                        break
                    old_rob=rob
                if it >= FLAGS.max_iter and rob >=max_rob:
                    break
                
                
                print("Robustness of student model after {} epochs of adversarial training (%) is {:.3f} with acc={:.3f}".format(epoch,rob,acc))
            
            print("Acc of student model {} after retraining (%): {:.3f}".format(i+1,acc))
            print("--- Adversarial training takes %s seconds ---" % (time.time() - start_time))
        # Save student models
        if new_train==False:
            dir_path = os.path.join(cwd,'experiments', FLAGS.data, FLAGS.data+"_models_"+''.join(str(FLAGS.lamda).split('.'))+'_'+str(FLAGS.n)+FLAGS.models_batch+'Xt')
        else:
            dir_path = os.path.join(cwd,'experiments', FLAGS.data,FLAGS.data+"_models_"+''.join(str(FLAGS.lamda).split('.'))+'_'+str(FLAGS.n)+'_'+FLAGS.models_batch)
        print(dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        filename = os.path.join(dir_path,"CNN_"+FLAGS.data + str(i+1) + ".pth")
        torch.save(model.state_dict(),filename)
        
    
def predict1(x):
    ''' predict the labels of a set x using conf-weighted scheduling of MTD'''
    
    
    models=[]
    for i in range(1,FLAGS.n+1):
        models.append(load_model(os.path.join(cwd,FLAGS.data+"_models_"+''.join(str(FLAGS.lamda).split('.'))+'_'+str(FLAGS.p)+'_'+FLAGS.models_batch),i))
    
    i=0
    y_probs=[]
    for model in models:
        if device == "cuda":
            model = model.cuda()
        model.eval()
        x=x.to(device)
        
        if i==0:
            y_probs = model(x)
        else:
            y_probs+= model(x)
        
        i+=1
    
    
    return y_probs


class Morphence():
    
    '''Morphence prediction system'''
    
    def __init__(self, test_size, Q_max,n,data,lamda,starting_batch, class_nb):
        
        self.test_size = test_size
        self.Q_max = Q_max
        self.n = n
        self.data = data
        self.lamda = lamda
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
        
        if self.Q_max<=5000:
            print('The distribution of {} queries with respect to Q_max = {} is {}.'.format(self.test_size,self.Q_max,self.queries))
        
    def predict2(self,x):
        ''' predict the labels of a set x using highest conf scheduling of MTD'''
        
        #print(self.nb_queries)
        print('Received {} queries'.format(x.shape[0]))
        
        # input transformations
        #trans_shift = 0.1+(random()*(0.25-0.1)) #scaled value = min + (value * (max - min))
        #rot_deg = 10+(random()*(20-10))
        #transform=torchvision.transforms.RandomAffine(degrees=0, translate=(trans_shift,trans_shift))
        #x=transform(x)
        
        # Gaussian Noise 
        #x = x + np.sqrt(0.1)*(0.1**0.5)*torch.randn(x.shape).to(device)
        
        y_probs=[] # prediction probabilities  of all models
        
        for qi in range(len(self.queries)-1):
            
            if self.nb_queries >= self.queries[qi] and self.nb_queries < self.queries[qi+1]:
                if self.nb_queries + x.shape[0] <= self.queries[qi+1]:
                    models=[]
                    for i in range(1,self.n+1):
                        try:
                            #models.append(load_model(os.path.join(cwd,self.data+"_models_"+''.join(str(self.lamda).split('.'))+'_'+str(FLAGS.p)),i))
                            models.append(load_model(os.path.join(cwd,'experiments',self.data,self.data+"_models_"+''.join(str(self.lamda).split('.'))+'_'+str(self.n)+'_'+self.starting_batch[0]+str(int(self.starting_batch[1])+qi)),i))
                        except FileNotFoundError:
                            raise('model {} is not found'.format(i))
                            
                    print('### Responding to queries from {} to {} using models pool {}'.format(self.nb_queries+1,self.nb_queries + x.shape[0],self.starting_batch[0]+str(int(self.starting_batch[1])+qi)))
                    for model in models:
                        if device == "cuda":
                            model = model.cuda()
                        model.eval()
                        y_probs.append(model(x))
                
                elif self.nb_queries + x.shape[0] > self.queries[qi+1]:
                    models1=[]
                    models2=[]
                    x1 = x[:self.queries[qi+1]-self.nb_queries]
                    print('### Responding to queries from {} to {} using models pool {}'.format(self.nb_queries+1,self.queries[qi+1],self.starting_batch[0]+str(int(self.starting_batch[1])+qi)))
                    if self.queries[qi+1] < self.test_size:
                        x2 = x[self.queries[qi+1]-self.nb_queries:]
                        print('And responding to queries from {} to {} using models pool {}'.format(self.queries[qi+1]+1,self.nb_queries + x.shape[0],self.starting_batch[0]+str(int(self.starting_batch[1])+qi+1)))
                    for i in range(1,self.n+1):
                        try:
                            models1.append(load_model(os.path.join(cwd,'experiments',self.data,self.data+"_models_"+''.join(str(self.lamda).split('.'))+'_'+str(self.n)+'_'+self.starting_batch[0]+str(int(self.starting_batch[1])+qi)),i))
                        except FileNotFoundError:
                            raise('model {} is not found'.format(i))
                            
                        if self.queries[qi+1] < self.test_size:
                            try:
                                models2.append(load_model(os.path.join(cwd,'experiments',self.data,self.data+"_models_"+''.join(str(self.lamda).split('.'))+'_'+str(self.n)+'_'+self.starting_batch[0]+str(int(self.starting_batch[1])+qi+1)),i))
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
                    y_ind = torch.reshape(y_probs[i][ind], (1, self.class_nb))
                else:
                    y_ind = torch.cat((y_ind,torch.reshape(y_probs[i][ind], (1, self.class_nb))),dim=0)
                
            #print('predictions of sample {} are {}'.format(ind,y_ind))
            #print('highest confidence vector of sample {} are {}'.format(ind, y_ind.max(0)[0]))
            
            
            #print('top k highest confidence vector of sample {} are {}'.format(ind, torch.topk(y_ind,5,dim=0)[0]))
            
            #print(y_ind.max(0))
            #print(y_ind.max(0)[0].argmax().item())
            selected_ind = y_ind.max(0)[1][y_ind.max(0)[0].argmax().item()].item()
            
            #if FLAGS.data=='CIFAR10' and selected_ind in [0,1,2]:
            #    selected_ind = torch.topk(y_ind,2,dim=0)[1][1][torch.topk(y_ind,2,dim=0)[0][1].argmax().item()]
            #print('selected model for sample {} is model {}'.format(ind, selected_ind))
            
            # keep track of selected models
            self.scheduling[selected_ind]+=1
            
            if ind == 0:
                y_pred = torch.reshape(y_ind.max(0)[0], (1, self.class_nb))
            else:
                y_pred = torch.cat((y_pred,torch.reshape(y_ind.max(0)[0], (1, self.class_nb))))
                
            
        
        
        return y_pred

def transferability(attack,data,size,batch_size=128):
    '''compute accuracy'''
    
    transf=[] # list of average transferabilities for each student model
    models=[]
    for i in range(1,FLAGS.n+1):
        models.append(load_model(os.path.join(cwd,FLAGS.data+"_models_"+''.join(str(FLAGS.lamda).split('.'))+'_'+str(FLAGS.p)),i))
    
    
    
    for i in range(len(models)):
        if device == "cuda":
            models[i] = models[i].cuda()
        models[i].eval()
        
        print('performing {} attack on model {}'.format(attack,i+1))
        x_adv, y_adv = perform_attack(models[i],data,size,attack=attack,model_type='student')
        
        transfi=[] # transferability of model i across all student models using using all adv data
        for j in range(len(models)):
            if j != i:
                if device == "cuda":
                    models[j] = models[j].cuda()
                models[j].eval()
                
                tot=0 # total of adv samples on model i
                s=0 # sum of transferable samples for model j
                for b_i in range(0,x_adv.shape[0],batch_size):
                    x, y = get_batch(x_adv,y_adv, b_i, batch_size)
                    x, y = x.to(device), y.to(device)
                    _, y_predi = models[i](x).max(1)
                    resi = y_predi.eq(y)
                    
                    _, y_predj = models[j](x).max(1)
                    resj = y_predj.eq(y)
                    
                    for ind in range(resi.shape[0]):
                        if resi[ind] == False: # evasion on model i
                            tot+=1
                            if resj[ind] == False: # transferable to model j
                                s+=1
                print('Transferability of model {} to model {}: {}'.format(i+1,j+1,float(s)/tot))
                transfi.append(float(s)/tot)
        transf.append(np.mean(transfi))
        print('Avergae Transferability of model {} across all models: {}'.format(i+1,np.mean(transfi)))
        
    print('Overall transferability of MTD framework using {} attack: {}'.format(attack,np.mean(transf)))
    
    return np.mean(transf)

def test_under_attack(model,data,size,attack='CW',batch_size=128,model_type='mtd',copycat=False):
    
    return robustness(model,data,size,batch_size=batch_size,attack=attack,model_type=model_type,copycat=copycat)

def generate_students(_):
    
    print("/*** Generating a batch of student models ***/\n")
    data = load_data()
    perturb_weights_and_retrain(cwd,data,FLAGS.lamda,FLAGS.n,FLAGS.p,batch_size=FLAGS.batch,new_train=True)

def test_base(_):
    
    print('Loading data ...')
    data = load_data()#train=False
    
    # test undefended master model without attack
    master_model=load_model(cwd,'')
    
    if FLAGS.attack=='NoAttack':
        print('Acc of master model ',accuracy(master_model,FLAGS.data,FLAGS.test_set,model_type='master'))
    else:
        print('Acc of MTD framework under {} attack {}'.format(FLAGS.attack,test_under_attack(master_model,FLAGS.data,FLAGS.test_set,attack=FLAGS.attack,model_type='master',copycat=False)))
    

def test_adv(_):
    
    print('Loading data ...')
    data = load_data()#train=False
    
    # test undefended master model without attack
    model=load_model(cwd,'')
    
    old_acc = accuracy(model,data,FLAGS.test_set,model_type='master')
    start_time = time.time()
    it = 0
    #if n-i <= p:
    print('# Performing adversarial training')
    epoch=0
    old_rob = robustness(model,data,1000,batch_size=FLAGS.batch,attack='FGS')
    max_rob = old_rob
    print("Robustness before adversarial training (%) is {:.3f}".format(old_rob))
    while True:
        epoch+=1
        model, acc = retrain(model,data,device,1,batch_size=FLAGS.batch,adversarial=True)
        rob = robustness(model,data,1000,batch_size=FLAGS.batch,attack='FGS')
        it +=1
        if rob > max_rob:
            max_rob = rob
        if rob < old_rob:
            old_rob=rob
        if epoch%5==0:
            print('old_rob',old_rob)
            if rob-old_rob<1:
                break
            old_rob=rob
        if it >= 50 and rob >=max_rob:
            break
    
    
    if FLAGS.attack=='NoAttack':
        print('Acc of master model ',accuracy(model,data,FLAGS.test_set,model_type='master_adv'))
    else:
        print('Acc of MTD framework under {} attack {}'.format(FLAGS.attack,test_under_attack(model,data,FLAGS.test_set,attack=FLAGS.attack,model_type='master_adv',copycat=False)))
        
def test(_):
    
    if FLAGS.Q_max>5000:
        raise('Q_max is higher than the test set size. use a lower value')
        
    if FLAGS.attack not in supported_attacks:
        raise('attack is not supported try CW, FGS or SPSA')
    print('Loading data ...')
    data = load_data()#train=False
    
    if FLAGS.attack == 'SPSA':
        dir_path=os.path.join(cwd,attack+'_test')
        if not os.path.exists(dir_path):
            mtd_inst = Morphence(FLAGS.6000*6000, 6000*6000,FLAGS.n,FLAGS.data,FLAGS.lamda,FLAGS.models_batch, FLAGS.class_nb)
            perform_attack(mtd_inst.predict2,data,FLAGS.test_set,attack='spsa',model_type='mtd',copycat=False)
         
    print('Initializing Morphence ...')
    mtd_inst = Morphence(FLAGS.test_set, FLAGS.Q_max,FLAGS.n,FLAGS.data,FLAGS.lamda,FLAGS.models_batch, FLAGS.class_nb)
    
    if FLAGS.attack=='NoAttack':
        print('Acc of MTD framework before attack',accuracy(mtd_inst.predict2,data,FLAGS.test_set,model_type='mtd'))
    else:
        print('Acc of MTD framework under {} attack {}'.format(FLAGS.attack, test_under_attack(mtd_inst.predict2,data,FLAGS.test_set,attack=FLAGS.attack,batch_size=FLAGS.batch,model_type='mtd',copycat=False)))  
