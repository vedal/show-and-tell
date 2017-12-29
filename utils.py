from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import time
import os

#plt.ion()   # interactive mode


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
    plt.show()

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

def load_models(checkpoint_file,sample=False):
    if sample:
        checkpoint = torch.load(checkpoint_file,map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(checkpoint_file)
    encoder_state_dict = checkpoint['encoder_state_dict']
    decoder_state_dict = checkpoint['decoder_state_dict']
    optimizer = checkpoint['optimizer']
    step = checkpoint['step']
    epoch = checkpoint['epoch']
    return encoder_state_dict, decoder_state_dict, optimizer, step, epoch

def dump_losses(losses_train, losses_val, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump({'losses_train': losses_train, 'losses_val': losses_val}, f)

def convert_back_to_text(idx_arr, vocab):
    sampled_caption = []
    for word_id in idx_arr:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break

    sentence = ' '.join(sampled_caption)
    return sentence

def sample(encoder, decoder, vocab, val_loader):
    encoder.batchnorm.eval()
    # run validation set
    images, captions, lengths = next(iter(val_loader))
    captions = to_var(captions, volatile=True)

    targets = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True)[0]
    features = encoder(to_var(images, volatile=True))

    # predict
    sampled_ids = decoder.sample(features)

    sampled_ids = sampled_ids.cpu().data.numpy()[0]
    predicted = convert_back_to_text(sampled_ids, vocab)

    true_ids = captions.cpu().data.numpy()[0]
    target = convert_back_to_text(true_ids, vocab)

    out = make_grid(images[0])
    imshow(out, figsize=(10,6), title='Target: %s\nPrediction: %s' % (target, predicted))

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
