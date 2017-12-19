from dataset_coco import *
from vocab import dump_vocab 

if __name__ == '__main__':
    download_coco()
    download_annotations()
    dump_vocab()



