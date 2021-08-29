# -*- coding: utf-8 -*-

#torch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#general
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
#local files:
from model import CNN
from cifar_data import get_datasets
from train_mnist import PyNet, ld_mnist
#system
from sys import argv, exit, stderr

if __name__ == '__main__':
    if len(argv) != 4:
        print('Use: {} model_filename.pth target_model.pth data_name'.format(argv[0]), file=stderr)
        exit(1)
    

    target_model_fn = argv[2]
    copycat_model_fn = argv[1]
    data_name = argv[3]
    batch_size = 128
    
    
    if data_name == 'CIFAR10':
        max_epochs = 30
        nb_train = 50000
    elif data_name == 'MNIST':
        max_epochs = 30
        nb_train = 60000
    else:
        raise('dataset {} is not supported: try {} or {}.'.format(data_name,'CIFAR10', 'MNIST'))
    # load target model and create copycat model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if target_model_fn == 'Morphence':
        from copycat_mtd import Morphence
        morph = Morphence(nb_train, int(nb_train/2) ,4,data_name,'b1', 10)
        target_model = morph.predict2
    else:
        if data_name == 'CIFAR10':
            target_model = CNN()
        elif data_name == 'MNIST':
            target_model = PyNet()
        target_model.load_state_dict(torch.load(target_model_fn))
        target_model = target_model.to(device)
        target_model.eval()
    
    
    
    if data_name == 'CIFAR10':
        trans_shift = 0.15
        rot_deg = 12
        transform = transforms.Compose([transforms.Resize( (32,32) ), 
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
                                        transforms.RandomAffine(degrees=rot_deg, translate=(trans_shift,trans_shift)) ])
        datasets = get_datasets(test=False, transform=transform, batch=batch_size, model='copycat')
        
    elif data_name == 'MNIST':
        trans_shift = 0.1
        rot_deg = 10
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomAffine(degrees=rot_deg, translate=(trans_shift,trans_shift)) ])
        datasets = ld_mnist(batch_size=batch_size, transform=transform,shuffle=False)
    
    
    # Stealing labels 
    print('Generating labels from oracle...')
    stolen_labels=[]
    total=0
    correct=0
    with torch.no_grad():
        shape=0
        with tqdm(datasets['train']) as tqdm_train:
            for data in tqdm_train:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                shape+=images.shape[0]
                outputs = target_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for label in predicted:
                    stolen_labels.append(label)
                    
    print('Accuracy: {:.2f}%'.format(100. * correct / total))
    
    print('Training Copycat model...')
    
    if data_name=='CIFAR10':
        criterion = nn.CrossEntropyLoss()
        copycat_model = CNN()
        optimizer = optim.SGD(copycat_model.parameters(), lr=1e-3, momentum=0.9)
        
    elif data_name=='MNIST':
        criterion = nn.CrossEntropyLoss(reduction="mean")
        copycat_model = PyNet()
        optimizer = optim.SGD(copycat_model.parameters(), lr=0.01, momentum=0.9)
        
    copycat_model = copycat_model.to(device)
    
    for epoch in range(max_epochs):
        running_loss = 0.0
        
        j=0
        with tqdm(datasets['train']) as tqdm_train:
            for i, data in enumerate(tqdm_train):
                
                inputs, _ = data
                inputs = inputs.to(device)
                labels = torch.LongTensor(stolen_labels[j:j+batch_size]).to(device)
                optimizer.zero_grad()
                outputs = copycat_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    
                running_loss += loss.item()
                if i % 200 == 199:
                    tqdm_train.set_description('Epoch: {}/{} Loss: {:.3f}'.format(
                        epoch+1, max_epochs, running_loss/200.))
                    running_loss = 0.0
                j+=batch_size
    print('Model trained.')
    print('Saving the model to "{}"'.format(copycat_model_fn))

    torch.save(copycat_model.state_dict(), copycat_model_fn)



