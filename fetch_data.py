"""
This requires the COCO API.

$ git clone https://github.com/cocodataset/cocoapi
$ cd cocoapi/PythonAPI/
$ make install
"""
from dataset_coco import *
from vocab import dump_vocab 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true',  help='Downloads the test set.')
    args = parser.parse_args()

    try:
        if args.test:
            download_coco(DataType.Test)
        else:
            download_coco(DataType.Train)
            download_coco(DataType.Val)
        download_annotations()
        dump_vocab()
    except Exception as e:
        print(e)
