import argparse
import torch.nn as nn
import torch
import SSD.data_import as data
import numpy as np
from sklearn.covariance import ledoit_wolf
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
import matplotlib.pyplot as plt
from torchvision import transforms
from random import randrange


from absl import app, flags


from SSD.model import SupResNet, SSLResNet


args = flags.FLAGS
device = "cuda:0"






def get_scores(ftrain, ftest, food, shrunkcov=False):
    if shrunkcov:
        print("Using ledoit-wolf covariance estimator.")
        cov = lambda x: ledoit_wolf(x)[0]
    else:
        cov = lambda x: np.cov(x.T, bias=True)

    # ToDO: Simplify these equations
    #tests the equation on a in distribution array
    dtest = np.sum(
        (ftest - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (ftest - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )


    dood = np.sum(
        (food - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (food - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )

    return dtest, dood
    



def create_load_model(args) :
        # create model
    if args.training_mode in ["SimCLR", "SupCon"]:
        #print(args.data)
        if args.data == 'CIFAR10':
            model = SSLResNet(arch=args.arch).eval()
        if args.data == 'MNIST':
            model = SSLResNet(arch=args.arch,in_channel=1).eval()#
    elif args.training_mode == "SupCE":
        model = SupResNet(arch=args.arch, num_classes=args.classes).eval()
    else:
        raise ValueError("Provide model class")
    model.encoder = nn.DataParallel(model.encoder).to(device)
    print('[MODEL]  model instanciated')

     # load checkpoint pretrained model
    ckpt_dict = torch.load(args.ckpt, map_location="cpu")
    if "model" in ckpt_dict.keys():
        ckpt_dict = ckpt_dict["model"]
    if "state_dict" in ckpt_dict.keys():
        ckpt_dict = ckpt_dict["state_dict"]
    model.load_state_dict(ckpt_dict)
    print('[CHECKPOINT] model checkpoint Loaded')
    return model

def get_features(model, dataloader,dimensions, max_images= 25000,verbose=False):
    features, labels = [], []
    total = 0
    model.eval()
    if(dimensions==2):
        
        for index ,(img, label) in enumerate(dataloader):
            if total > max_images:
                break
            img, label = img.cuda(), label.cuda()
            '''
            if attack=="FGSM":
                img = fast_gradient_method(model, img, eps=0.2, norm = np.inf)
            if attack =="CW":
                img = carlini_wagner_l2(model, img, 10,label,targeted = False)
            '''
            features += list(model(img).data.cpu().numpy())
            labels += list(label.data.cpu().numpy())

            total += len(img) 
        return np.array(features), np.array(labels)
    elif(dimensions == 1) :
        for index,img in enumerate(dataloader):
            if total > max_images:
                break
            img = img.cuda()
            '''
            if attack=="FGSM":
                img = fast_gradient_method(model, img, eps=0.2, norm = np.inf)
            if attack =="CW":
                img = carlini_wagner_l2(model, img, 10,label,targeted = False)
            '''
            features += list(model(img).data.cpu().numpy())

            total += len(img) 
        return np.array(features)
  
        

 

def dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])  


def classify_ssd (x, model,train_loader,test_loader):
   


    
  
    
    # pass train data on pre trained model    
    features_train, labels_train = get_features(
            model.encoder, train_loader,dimensions=2,max_images=1000
        )  # using feature befor MLP-head
        
    
    features_test, _ = get_features(model.encoder, test_loader,dimensions=2,max_images=1000,verbose =False)


    
    transform_test = [transforms.Resize(args.size), transforms.ToTensor()]
    transform_test = transforms.Compose(transform_test)
    x.transform = transform_test
    

    ood_loader = torch.utils.data.DataLoader (x,args.batch)
    
    

    x_features = get_features(model.encoder ,ood_loader,dimensions=1,max_images=250000,verbose=False)
  

    

    # TODO Test the resize 
    score_test,score_ood = get_scores(features_train,features_test,x_features)

    index= randrange(1,score_test.size)
    score_test = score_test[index:index+args.batch]


    #print(score_test)
    #print(score_ood)


    N_ood = np.arange(0,len(score_ood))
    N_test = np.arange(0,len(score_test))
    
    '''
    fig = plt.figure(figsize=(10,10))
    a = fig.add_subplot()
    
    a.scatter(N_ood,score_ood, s=1, c='r', marker="s" )
    a.scatter(N_test,score_test, s=1, c='g', marker="o")        
    a.legend(title="DataSets")
    plt.savefig('./scheduling.pdf')
    #plt.show()
    '''
    threshhold = np.percentile(score_test,95)

    if(np.mean(score_ood)>threshhold):
        return True
    else :
        return False
    
    


if __name__ =="__main__":
    classify_ssd()