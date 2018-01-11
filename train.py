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
from data_loader import get_coco_data_loader
from models import CNN, RNN
from vocab import Vocabulary, load_vocab
import os

def main(args):
    # hyperparameters
    batch_size = args.batch_size
    num_workers = 1

    # Image Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    # load COCOs dataset
    IMAGES_PATH = 'data/train2014'
    CAPTION_FILE_PATH = 'data/annotations/captions_train2014.json'

    vocab = load_vocab()
    train_loader = get_coco_data_loader(path=IMAGES_PATH,
                                        json=CAPTION_FILE_PATH,
                                        vocab=vocab,
                                        transform=transform,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers)

    IMAGES_PATH = 'data/val2014'
    CAPTION_FILE_PATH = 'data/annotations/captions_val2014.json'
    val_loader = get_coco_data_loader(path=IMAGES_PATH,
                                      json=CAPTION_FILE_PATH,
                                      vocab=vocab,
                                      transform=transform,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)


    losses_val = []
    losses_train = []

    # Build the models
    ngpu = 1
    initial_step = initial_epoch = 0
    embed_size = args.embed_size
    num_hiddens = args.num_hidden
    learning_rate = 1e-3
    num_epochs = 3
    log_step = args.log_step
    save_step = 500
    checkpoint_dir = args.checkpoint_dir

    encoder = CNN(embed_size)
    decoder = RNN(embed_size, num_hiddens, len(vocab), 1, rec_unit=args.rec_unit)

    # Loss
    criterion = nn.CrossEntropyLoss()

    if args.checkpoint_file:
        encoder_state_dict, decoder_state_dict, optimizer, *meta = utils.load_models(args.checkpoint_file,args.sample)
        initial_step, initial_epoch, losses_train, losses_val = meta
        encoder.load_state_dict(encoder_state_dict)
        decoder.load_state_dict(decoder_state_dict)
    else:
        params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.batchnorm.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    if args.sample:
        return utils.sample(encoder, decoder, vocab, val_loader)

    # Train the Models
    total_step = len(train_loader)
    try:
        for epoch in range(initial_epoch, num_epochs):

            for step, (images, captions, lengths) in enumerate(train_loader, start=initial_step):

                # Set mini-batch dataset
                images = utils.to_var(images, volatile=True)
                captions = utils.to_var(captions)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                
                # Forward, Backward and Optimize
                decoder.zero_grad()
                encoder.zero_grad()

                if ngpu > 1:
                    # run on multiple GPU
                    features = nn.parallel.data_parallel(encoder, images, range(ngpu))
                    outputs = nn.parallel.data_parallel(decoder, features, range(ngpu))
                else:
                    # run on single GPU
                    features = encoder(images)
                    outputs = decoder(features, captions, lengths)

                train_loss = criterion(outputs, targets)
                losses_train.append(train_loss.data[0])
                train_loss.backward()
                optimizer.step()

                # Run validation set and predict
                if step % log_step == 0:
                    encoder.batchnorm.eval()
                    # run validation set
                    batch_loss_val = []
                    for val_step, (images, captions, lengths) in enumerate(val_loader):
                        images = utils.to_var(images, volatile=True)
                        captions = utils.to_var(captions, volatile=True)

                        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                        features = encoder(images)
                        outputs = decoder(features, captions, lengths)
                        val_loss = criterion(outputs, targets)
                        batch_loss_val.append(val_loss.data[0])

                    losses_val.append(np.mean(batch_loss_val))

                    # predict
                    sampled_ids = decoder.sample(features)
                    sampled_ids = sampled_ids.cpu().data.numpy()[0]
                    sentence = utils.convert_back_to_text(sampled_ids, vocab)
                    print('Sample:', sentence)

                    true_ids = captions.cpu().data.numpy()[0]
                    sentence = utils.convert_back_to_text(true_ids, vocab)
                    print('Target:', sentence)

                    print('Epoch: {} - Step: {} - Train Loss: {} - Eval Loss: {}'.format(epoch, step, losses_train[-1], losses_val[-1]))
                    encoder.batchnorm.train()

                # Save the models
                if (step+1) % save_step == 0:
                    utils.save_models(encoder, decoder, optimizer, step, epoch, losses_train, losses_val, checkpoint_dir)
                    utils.dump_losses(losses_train, losses_val, os.path.join(checkpoint_dir, 'losses.pkl'))

    except KeyboardInterrupt:
        pass
    finally:
        # Do final save
        utils.save_models(encoder, decoder, optimizer, step, epoch, losses_train, losses_val, checkpoint_dir)
        utils.dump_losses(losses_train, losses_val, os.path.join(checkpoint_dir, 'losses.pkl'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', type=str,
            default=None, help='path to saved checkpoint')
    parser.add_argument('--checkpoint_dir', type=str,
            default='checkpoints', help='directory to save checkpoints')
    parser.add_argument('--batch_size', type=int,
            default=128, help='size of batches')
    parser.add_argument('--rec_unit', type=str,
            default='gru', help='choose "gru", "lstm" or "elman"')
    parser.add_argument('--sample', default=False, 
            action='store_true', help='just show result, requires --checkpoint_file')
    parser.add_argument('--log_step', type=int,
            default=125, help='number of steps in between calculating loss')
    parser.add_argument('--num_hidden', type=int,
            default=512, help='number of hidden units in the RNN')
    parser.add_argument('--embed_size', type=int,
            default=512, help='number of embeddings in the RNN')
    args = parser.parse_args()
    main(args)
