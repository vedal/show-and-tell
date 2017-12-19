"""
This requires the COCO API.

$ git clone https://github.com/cocodataset/cocoapi
$ cd cocoapi/PythonAPI/
$ make
"""

from pycocotools.coco import COCO
from dataset_utils import download_and_extract
import os
from enum import Enum

PATH_TO_DATA = 'data'

class DataType(Enum):
    Train = 'train'
    Val = 'val'
    Test = 'test'

def path_to_imgs(dataType=DataType.Val):
    return 'data/{}2017'.format(dataType)

def path_to_captions(dataType=DataType.Val):
    return 'data/annotations/captions_{}2017.json'.format(dataType.value)

def download_coco(dataType=DataType.Val, dataDir=PATH_TO_DATA):
    baseURL = 'http://images.cocodataset.org/zips/{}2017.zip'
    url = baseURL.format(dataType.value)
    download_and_extract(url, dst=dataDir)
    return os.path.join(dataDir, dataType.value)

def download_annotations(dataDir=PATH_TO_DATA):
    annotationsURL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    download_and_extract(annotationsURL, dst=dataDir)

    dataTypes = [DataType.Train, DataType.Val]
    annFile = '{}/annotations/captions_{}2017.json'
    annFiles = map(lambda dataType: annFile.format(dataDir, dataType.value), dataTypes)
    return [annFile for annFile in annFiles]
