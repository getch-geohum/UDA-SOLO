from .registry import DATASETS
from .coco import CocoDataset

@DATASETS.register_module
class MyDataset(CocoDataset):
    CLASSES = ("background", "dwelling",) # "background"

