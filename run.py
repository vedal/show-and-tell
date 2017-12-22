"""
This requires the COCO API.

$ git clone https://github.com/cocodataset/cocoapi
$ cd cocoapi/PythonAPI/
$ make install
"""

from dataset_coco import *
from vocab import dump_vocab 

if __name__ == '__main__':
    download_coco()
    download_annotations()
    dump_vocab()
