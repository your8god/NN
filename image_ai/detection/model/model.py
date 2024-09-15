""" Model files """

from enum import Enum


class Model(Enum):
    """ Filenames for models """
    RetinaNet = 'retinanet_resnet50_fpn_coco-eeacb38b.pth'
    YOLOv3 = 'yolov3.pt'
    TinyYOLOv3 = 'tiny-yolov3.pt'
    
    @property
    def path(self):
        return r'./image_ai/detection/model/storage/' + self.value
    