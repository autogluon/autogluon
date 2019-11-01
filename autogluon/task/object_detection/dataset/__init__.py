from ....core import *
from .base import *
from .voc import *


@autogluon_function()
def get_dataset(dataset_name, dataset_path):
    if dataset_name=="voc":
        print("voc")
        return VOC
    elif dataset_name=="coco":
        print("coco")
        return VOC