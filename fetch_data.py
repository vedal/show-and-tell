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
    parser.add_argument('--dataType', type=str,
            default='train', help='type of dataset to download: train/val/test')
    args = parser.parse_args()

    try:
        dataType = DataType(args.dataType)
        download_coco(dataType)
        download_annotations()
        dump_vocab()
    except Exception as e:
        print(e)
