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

def main():
    # hyperparameters
    batch_size = 8
    num_workers = 1

    # Image Preprocessing
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transforms = {
        'train': transforms.Compose([
            transforms.Scale(256), # Original network trained on ImageNet 256x256
            #transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    }

    # load COCOs dataset
    IMAGES_PATH = 'data/val2017'
    CAPTION_FILE_PATH = 'data/annotations/captions_val2017.json'
    train_loader = get_coco_data_loader(path=IMAGES_PATH,
                                        json=CAPTION_FILE_PATH,
                                        vocab=None, # TODO, Wojtek
                                        transform=transforms,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers)


    # show some sample images
    (images, labels) = next(iter(train_loader))
    #out = make_grid(images)
    #utils.imshow(out,figsize=(10,15),title=[label_names[x] for x in labels])


    images = Variable(images)
    labels = Variable(labels)


    # load pretrained ResNet18 model
    # TODO: consider using DenseNet instead
    # https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
    original_model = models.resnet18(pretrained=True)
    # TODO: re-write as its own class according to the pytorch tutorials, with proper forward()
    # TODO: this is needed for having variable output_dim
    #model = utils.FeatureExtractor(original_model, output_dim=1001)

    # freeze weights
    for param in model.parameters():
        param.requires_grad = False

    outputs = model(images) # batch_size x 1000

    print model

if __name__ == '__main__':
    main()