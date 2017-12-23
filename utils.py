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


def save_models(encoder, decoder, optimizer, epoch, step, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_file = os.path.join(checkpoint_path, 'model-%d-%d.ckpt' %(epoch+1, step+1))
    print('Saving model to:', checkpoint_file)
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'epoch': epoch,
        'step': step,
        'optimizer': optimizer
        }, checkpoint_file)

def load_models(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    encoder_state_dict = checkpoint['encoder_state_dict']
    decoder_state_dict = checkpoint['decoder_state_dict']
    optimizer = checkpoint['optimizer']
    step = checkpoint['step']
    epoch = checkpoint['epoch']
    return encoder_state_dict, decoder_state_dict, optimizer, step, epoch
