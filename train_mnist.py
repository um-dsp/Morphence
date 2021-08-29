# -*- coding: utf-8 -*-
"""

@author: Abderrahmen Amich
@email:  aamich@umich.edu
"""


from absl import app, flags
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
import torch.optim as optim

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

import os
FLAGS = flags.FLAGS

# Get current working directory
cwd = os.getcwd()

class PyNet(nn.Module):
    """CNN architecture. This is the same MNIST model from pytorch/examples/mnist repository"""

    def __init__(self, in_channels=1):
        super(PyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def ld_mnist(batch_size=128, transform=None,shuffle=True):
    """Load training and test data."""
    
    if transform==None:
        train_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )
        test_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )
    else:
        train_transforms = transform
        
        test_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )

    # Load MNIST dataset
    train_dataset = MNIST(root='./data', train=True, download=True, transform=train_transforms)
    test_dataset = MNIST(root='./data', train=True, download=True, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return EasyDict(train=train_loader, test=test_loader)


def main(_):
    # Load training and test data
    data = ld_mnist()

    # Instantiate model, loss, and optimizer for training
    net = PyNet(in_channels=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train model
    net.train()
    for epoch in range(1, FLAGS.nb_epochs + 1):
        train_loss = 0.0
        for x, y in data.train:
            x, y = x.to(device), y.to(device)
            '''
            if FLAGS.adv_train:
                # Replace clean example with adversarial example for adversarial training
                x = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)
            '''
            optimizer.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(
            "epoch: {}/{}, train loss: {:.3f}".format(
                epoch, FLAGS.nb_epochs, train_loss
            )
        )

    # Evaluate on clean and adversarial data
    net.eval()
    report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
    for x, y in data.test:
        x, y = x.to(device), y.to(device)
        
        _, y_pred = net(x).max(1)  # model prediction on clean examples
        report.nb_test += y.size(0)
        report.correct += y_pred.eq(y).sum().item()
        
        
    print("test acc on clean examples (%): {:.3f}".format(report.correct / report.nb_test * 100.0))
    
    # save model
    filename = os.path.join(cwd,"CNN_MNIST.pth")
    torch.save(net.state_dict(),filename)
    

if __name__ == "__main__":
    flags.DEFINE_integer("nb_epochs", 8, "Number of epochs.")

    app.run(main)