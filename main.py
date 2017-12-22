from __future__ import print_function
import torch
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid
from torchvision import transforms
from torch import np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import utils
from data_loader import get_coco_data_loader
from models import CNN
from vocab import Vocabulary, load_vocab

def main():
    # hyperparameters
    batch_size = 8
    num_workers = 1
    cnn_output_dim = 1001

    # Image Preprocessing
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transforms = {
        'train': transforms.Compose([
            transforms.Scale(256),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), 
                                     (0.229, 0.224, 0.225))
        ])
    }

    # load CIFAR10 dataset
    """
    DATA_PATH = './data/CIFAR10/'
    train_dataset = datasets.CIFAR10(root=DATA_PATH,
                                     train=True,
                                     transform=data_transforms['train'],
                                     download=True)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    # take one batch of images
    images, _ = next(iter(train_loader))

    model = CNN(cnn_output_dim)

    images = Variable(images)

    # forward-pass
    outputs = model(images)

    print(model)
    print(outputs.size()) # batch_size x cnn_output_dim


    """
    # load COCOs dataset
    IMAGES_PATH = 'data/val2017'
    CAPTION_FILE_PATH = 'data/annotations/captions_val2017.json'

    vocab = load_vocab()
    train_loader = get_coco_data_loader(path=IMAGES_PATH,
                                        json=CAPTION_FILE_PATH,
                                        vocab=vocab,
                                        transform=data_transforms['train'],
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers)


    # show some sample images
    images, captions, lengths = next(iter(train_loader))
    out = make_grid(images[0])
    utils.imshow(out, figsize=(10,6), title=[vocab.idx2word[idx] for idx in captions[0]])

    input('Press any button to continue...')
    images = Variable(images)
    labels = Variable(captions)

    # load pretrained ResNet18 model
    original_model = models.resnet18(pretrained=True)
    # TODO: re-write as its own class according to the pytorch tutorials, with proper forward()
    # TODO: this is needed for having variable output_dim
    #model = utils.FeatureExtractor(original_model, output_dim=1001)

    # freeze weights
    for param in original_model.parameters():
        param.requires_grad = False

    outputs = original_model(images) # batch_size x 1000

    print(original_model)

    #"""

if __name__ == '__main__':
    main()
