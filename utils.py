# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os

plt.ion()   # interactive mode



class FeatureExtractor(nn.Module):
    """Class to build new model including all but last layers"""
    def __init__(self, original_model,output_dim=1000):
        super(FeatureExtractor, self).__init__()

        from torch.nn import Sequential
        self.features = Sequential(
            # stop at conv4
            *list(original_model.children())[:-2] + [nn.Linear(8, output_dim)]
        )
    def forward(self, x):
        x = self.features(x)
        return x


def imshow(inp, figsize=None, title=None):
    if figsize != None:
        plt.figure(figsize=figsize)
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict