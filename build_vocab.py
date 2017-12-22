import nltk
import pickle
from collections import Counter
from pycocotools.coco import COCO


class VocabWrapper(object):
    # Wrapper for dictionaries from pytorch tutorial 03
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.count = 0

    def add_word(self,word):
        # add new words to dict
        if not word in self.word2id:
            self.word2id[word] = count
            self.id2word[count] = word
            self.count += 1

    def __call__(self,word):
        # look up id of word; for encoding words
        if not word in self.word2id:
            return self.word2id['<UNK>']
        return self.word2id[word]



json = 'data/annotations/captions_val2017.json'
threshold = 5

coco = COCO(json)
counter = Counter()
ids = coco.anns.keys()
for i, id in enumerate(ids):
    caption = str(coco.anns[id]['caption'])
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    counter.update(tokens)

words = [word for word, count in counter.items() if count >= threshold]

vocab = VocabWrapper()
vocab.add_word('<START>')
vocab.add_word('<EOS>')
vocab.add_word('<PAD>')
vocab.add_word('<UNK>')

for word in words:
    vocab.add_word(word)
