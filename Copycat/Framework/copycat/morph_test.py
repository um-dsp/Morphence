# -*- coding: utf-8 -*-

"""

@author: Abderrahmen Amich
@email:  aamich@umich.edu
"""


#torch
import torch
#local files
from model import CNN
from cifar_data import get_datasets
from copycat_mtd import Morphence
#general
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
#system
from sys import argv, exit, stderr

if __name__ == '__main__':
    if len(argv) != 2:
        print('Use: {} model_file.pth'.format(argv[0]), file=stderr)
        exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    morph = Morphence(10000, 5000,4,'CIFAR10','b3', 10)
    model = CNN()
    target = morph.predict2
    model.load_state_dict(torch.load(argv[1]))
    model.to(device)
    
    batch_size = 128
    dataset = get_datasets(train=False, batch=batch_size)

    print('Testing the model...')
    correct = 0
    matching = 0
    total = 0
    #results = np.zeros([dataset['n_test'], 2], dtype=np.int)
    res_pos = 0
    with torch.no_grad():
        model.eval()
        #target.eval()
        for data in tqdm(dataset['test']):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs_target = target(images)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted_target = torch.max(outputs_target.data, 1)
            #results[res_pos:res_pos+batch_size, :] = np.array([labels.tolist(), predicted.tolist()]).T
            res_pos += batch_size
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            matching += (predicted_target == predicted).sum().item()
            

    #micro_avg = f1_score(results[:,0], results[:,1], average='micro')
    #macro_avg = f1_score(results[:,0], results[:,1], average='macro')
    #print('\nAverage: {:.2f}% ({:d} images)'.format(100. * (correct/total), total))
    #print('Micro Average: {:.6f}'.format(micro_avg))
    #print('Macro Average: {:.6f}'.format(macro_avg))

    print('Accuracy: {:.2f}%'.format(100. * correct / total))
    print('Fidelity: {:.2f}%'.format(100. * matching / total))



