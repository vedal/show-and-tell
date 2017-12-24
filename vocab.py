import nltk
import pickle
import argparse
import os
from collections import Counter
from pycocotools.coco import COCO
from dataset_coco import PATH_TO_DATA

def path_to_vocab():
    return os.path.join(PATH_TO_DATA, 'vocab.pkl')


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def start_token(self):
        return '<start>'

    def end_token(self):
        return '<end>'


def build_vocab(json='data/annotations/captions_train2017.json', threshold=4, max_words=15000):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." %(i, len(ids)))

    # 4 special tokens
    words = counter.most_common(max_words-4)
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in words if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word(vocab.start_token())
    vocab.add_word(vocab.end_token())
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    print('Total number of words in vocab:', len(words))
    return vocab

def dump_vocab(path=path_to_vocab()):
    if not os.path.exists(path):
        vocab = build_vocab()
        with open(path, 'wb') as f:
            pickle.dump(vocab, f)
        print("Total vocabulary size: %d" %len(vocab))
        print("Saved the vocabulary wrapper to '%s'" %path)
    else:
        print('Vocabulary already exists.')

def load_vocab(path=path_to_vocab()):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError('Failed to load %s: %s' % (path, e))


def main(args):
    vocab = build_vocab(json=args.caption_path,
                        threshold=args.threshold)
    vocab_path = args.vocab_path
    dump_vocab(vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default=path_to_vocab(),
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
