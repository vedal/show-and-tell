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

def save_models(encoder, decoder, optimizer, step, epoch, losses_train, losses_val, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_file = os.path.join(checkpoint_path, 'model-%d-%d.ckpt' %(epoch+1, step+1))
    print('Saving model to:', checkpoint_file)
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer': optimizer,
        'step': step,
        'epoch': epoch,
        'losses_train': losses_train,
        'losses_val': losses_val
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
    losses_train = checkpoint['losses_train']
    losses_val = checkpoint['losses_val']
    return encoder_state_dict, decoder_state_dict, optimizer, step, epoch, losses_train, losses_val

def dump_losses(losses_train, losses_val, path):
    import pickle
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(path, 'wb') as f:
        try:
            pickle.dump({'losses_train': losses_train, 'losses_val': losses_val}, f, protocol=2)
        except:
            pickle.dump({'losses_train': losses_train, 'losses_val': losses_val}, f)

def convert_back_to_text(idx_arr, vocab):
    from itertools import takewhile
    blacklist = [vocab.word2idx[word] for word in [vocab.start_token()]]
    predicate = lambda word_id: vocab.idx2word[word_id] != vocab.end_token()
    sampled_caption = [vocab.idx2word[word_id] for word_id in takewhile(predicate, idx_arr) if word_id not in blacklist]

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
    print('Target: ', target)
    print('Prediction: ', predicted)
    imshow(out, figsize=(10,6), title='Target: %s\nPrediction: %s' % (target, predicted))

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
