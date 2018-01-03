from __future__ import print_function
import torch
from torchvision import datasets, models, transforms
from torchvision import transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim
import torch.nn as nn
from torch import np
import utils
from data_loader import get_coco_data_loader, get_basic_loader
from models import CNN, RNN
from vocab import Vocabulary, load_vocab
import os

def main(args):
    # hyperparameters
    batch_size = args.batch_size
    num_workers = 2

    # Image Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    vocab = load_vocab()

    loader = get_basic_loader(dir_path=os.path.join(args.image_path),
                              transform=transform,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)


    # Build the models
    embed_size = args.embed_size
    num_hiddens = args.num_hidden
    checkpoint_path = 'checkpoints'

    encoder = CNN(embed_size)
    decoder = RNN(embed_size, num_hiddens, len(vocab), 1, rec_unit=args.rec_unit)

    encoder_state_dict, decoder_state_dict, optimizer, *meta = utils.load_models(args.checkpoint_file)
    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Train the Models
    try:
        results = []
        for step, (images, image_ids) in enumerate(loader):
            images = utils.to_var(images, volatile=True)

            features = encoder(images)
            captions = decoder.sample(features)
            captions = captions.cpu().data.numpy()
            captions = [utils.convert_back_to_text(cap, vocab) for cap in captions]
            captions_formatted = [{'image_id': int(img_id), 'caption': cap} for img_id, cap in zip(image_ids, captions)]
            results.extend(captions_formatted)
            print('Sample:', captions_formatted)
    except KeyboardInterrupt:
        print('Ok bye!')
    finally:
        import json
        file_name = 'captions_model.json'
        with open(file_name, 'w') as f:
            json.dump(results, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', type=str,
            default=None, help='path to saved checkpoint')
    parser.add_argument('--batch_size', type=int,
            default=128, help='size of batches')
    parser.add_argument('--rec_unit', type=str,
            default='gru', help='choose "gru", "lstm" or "elman"')
    parser.add_argument('--image_path', type=str,
            default='data/test2014', help='path to the directory of images')
    parser.add_argument('--embed_size', type=int,
            default='512', help='number of embeddings')
    parser.add_argument('--num_hidden', type=int,
            default='512', help='number of embeddings')
    args = parser.parse_args()
    main(args)
