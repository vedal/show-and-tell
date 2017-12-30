import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
import re
from vocab import Vocabulary
from PIL import Image
from pycocotools.coco import COCO

class CocoDataset(data.Dataset):
    """Coco custom dataset class, compatible with Dataloader
       inspired by pytorch tutorial 03-advanced"""
    def __init__(self, path, json, vocab=None, transform=None):
        """

        :param path: images directory path
        :param json: annotations file path (annotations contain object instances, object keypoints and image captions)
                     this is the Common Objects Context (COCO)
        :param vocab: vocabulary
        :param transform: a torchvision.transforms image transformer for preprocessing
        """
        self.path = path
        self.coco = COCO(json) # object of Coco Helper Class
        self.ids = list(self.coco.anns.keys()) # unique identifiers for annontations
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self,index):
        """special python method for indexing a dict. dict[index]
        helper method to get annotation by id from coco.anns

        :param index: desired annotation id (key of annotation dict)

        return: (image, caption)
        """

        coco = self.coco
        vocab = self.vocab
        annotation_id = self.ids[index]
        caption = coco.anns[annotation_id]['caption']
        image_id = coco.anns[annotation_id]['image_id']
        path = coco.loadImgs(image_id)[0]['file_name']

        image = Image.open(os.path.join(self.path, path))
        image = image.convert('RGB')

        if self.transform != None:
            # apply image preprocessing
            image = self.transform(image)

        # tokenize captions
        caption_str = str(caption).lower()
        tokens = nltk.tokenize.word_tokenize(caption_str)
        caption = torch.Tensor([vocab(vocab.start_token())] +
                               [vocab(token) for token in tokens] +
                               [vocab(vocab.end_token())])

        return image, caption

    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    """Create mini-batches of (image, caption)

    Custom collate_fn for torch.utils.data.DataLoader is necessary for patting captions

    :param data: list; (image, caption) tuples
            - image: tensor;    3 x 256 x 256
            - caption: tensor;  1 x length_caption

    Return: mini-batch
    :return images: tensor;     batch_size x 3 x 256 x 256
    :return padded_captions: tensor;    batch_size x length
    :return caption_lengths: list;      lenghths of actual captions (without padding)
    """

    # sort data by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge image tensors (stack)
    images = torch.stack(images, 0)

    # Merge captions
    caption_lengths = [len(caption) for caption in captions]

    # zero-matrix num_captions x caption_max_length
    padded_captions = torch.zeros(len(captions), max(caption_lengths)).long()

    # fill the zero-matrix with captions. the remaining zeros are padding
    for ix, caption in enumerate(captions):
        end = caption_lengths[ix]
        padded_captions[ix, :end] = caption[:end]
    return images, padded_captions, caption_lengths

def get_coco_data_loader(path, json, vocab, transform=None,
        batch_size=32, shuffle=True, num_workers=2):
    """Returns custom COCO Dataloader"""

    coco = CocoDataset(path=path,
                       json=json,
                       vocab=vocab,
                       transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)

    return data_loader


class ImagesDataset(data.Dataset):
    """
    """
    def __init__(self, dir_path, transform=None):
        """
        :param dir_path:
        :param transform:
        :returns:
        """
        self.dir_path = dir_path
        self.transform = transform

        _a, _b, files = next(os.walk(dir_path))
        self.file_names = files

    def __getitem__(self, idx):
        """
        :param idx:
        :returns:
        """
        # load image and caption
        file_name = self.file_names[idx]
        image_id = re.findall('[0-9]{12}', file_name)[0]
        image_path = os.path.join(self.dir_path, file_name)
        image = Image.open(image_path).convert('RGB')

        # transform image
        if self.transform is not None:
            image = self.transform(image)

        return image, image_id

    def __len__(self):
        """
        :returns:
        """
        return len(self.file_names)

def get_basic_loader(dir_path, transform, batch_size=32, shuffle=True, num_workers=2):
    """
    Returns torch.utils.data.DataLoader for custom coco dataset.
    :param dir_path:
    :param ann_path:
    :param vocab:
    :param transform:
    :param batch_size:
    :param shuffle:
    :param num_workers:
    :returns:
    """
    datas = ImagesDataset(dir_path=dir_path, transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = data.DataLoader(dataset=datas, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers)
    return data_loader
