import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
#from build_vocab import Vocabulary
from pycocotools.coco import COCO

class CocoDataset(data.Dataset):
    """Coco custom dataset class, compatible with Dataloader
       inspired by pytorch tutorial 03-advanced"""
    def __init__(self, path, json, vocab=None, transform=None): # TODO: vocab
        """

        :param path: images directory path
        :param json: annotations file path (annotations contain object instances, object keypoints and image captions)
                     this is the Common Objects Context (COCO)
        :param vocab: vocabulary
        :param transform: a tourchvision.transforms image transformer for preprocessing
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

        image = Image.open(os.path.join(self.root, path)).convert('RBG')
        if self.transform != None:
            # apply image preprocessing
            image = self.transform(image)

        # tokenize captions
        caption_str = str(caption).lower()
        tokens = nltk.tokenize.word_tokenize(caption_str)
        caption = torch.Tensor([vocab('<start>')] +
                               [vocab(token) for token in tokens] +
                               [vocab('<eos>')])

        return image, caption

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

def get_coco_data_loader(path, json, vocab, transform, batch_size, shuffle, num_workers):
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




"""
def main():
    image_dir = 'data/val2017'
    caption_path = 'data/annotations/captions_val2017.json'
    CocoDataset(image_dir, caption_path)

main()
"""