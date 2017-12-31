import os
from pycocotools.coco import COCO
from downloader import download_and_extract
from enum import Enum

PATH_TO_DATA = 'data'

class DataType(Enum):
    Train = 'train'
    Val = 'val'
    Test = 'test'

def path_to_imgs(dataType=DataType.Train):
    return 'data/{}2014'.format(dataType)

def path_to_captions(dataType=DataType.Train):
    return 'data/annotations/captions_{}2014.json'.format(dataType.value)

def download_coco(dataType=DataType.Train, dataDir=PATH_TO_DATA):
    baseURL = 'http://images.cocodataset.org/zips/{}2014.zip'
    url = baseURL.format(dataType.value)
    download_and_extract(url, dst=dataDir)
    return os.path.join(dataDir, dataType.value)

def download_annotations(dataDir=PATH_TO_DATA):
    annotationsURL = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
    download_and_extract(annotationsURL, dst=dataDir)

    dataTypes = [DataType.Train, DataType.Train]
    annFile = '{}/annotations/captions_{}2014.json'
    annFiles = map(lambda dataType: annFile.format(dataDir, dataType.value), dataTypes)
    return [annFile for annFile in annFiles]
