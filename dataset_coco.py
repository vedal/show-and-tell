"""
This requires the COCO API.

$ git clone https://github.com/cocodataset/cocoapi
$ cd cocoapi/PythonAPI/
$ make
"""

from pycocotools.coco import COCO
from dataset_utils import download_and_extract
import os


def coco(dataDir='data', dataType='train2017'):
  baseURL = 'http://images.cocodataset.org/zips/{}.zip'
  url = baseURL.format(dataType)
  download_and_extract(url, dst=dataDir)
  return os.path.join(dataDir, dataType)

def annotations(dataDir='data'):
  annotationsURL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
  download_and_extract(annotationsURL, dst=dataDir)

  dataTypes = ['train2017', 'val2017']
  annFile = '{}/annotations/captions_{}.json'
  annFiles = map(lambda dataType: annFile.format(dataDir, dataType), dataTypes)
  return [annFile for annFile in annFiles]

if __name__ == '__main__':
    # Downloads COCO and fetches captions
    dataDir = 'data'

    # change to train etc if we want lots of data
    root_dir = coco(dataType='val2017')
    annFile_train, annFile_val = annotations(dataDir)
    train_dataset = datasets.CocoCaptions(root=root_dir,
                                          annFile=annFile_val,
                                          transform=data_transforms['train']) 
