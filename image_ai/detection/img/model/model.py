""" Model files """

from enum import Enum


class Model(Enum):
    """ Filenames for models """
    RetinaNet = 'retinanet_resnet50_fpn_coco-eeacb38b.pth'
    YOLOv3 = 'yolov3.pt'
    TinyYOLOv3 = 'tiny-yolov3.pt'
    
    @property
    def path(self):
        return r'./image_ai/detection/img/model/storage/' + self.value
    
    def __str__(self):
        return self.value.split('_')[0].split('-')[0].upper()
    