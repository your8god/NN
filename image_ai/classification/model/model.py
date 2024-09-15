""" Model files """

from enum import Enum


class Model(Enum):
    """ Filenames for models """
    MobileNetV2 = 'mobilenet_v2-b0353104.pth'
    RestNet50 = 'resnet50-19c8e357.pth'
    InceptionV3 = 'inception_v3_google-1a9a5a14.pth'
    DenseNet121 = 'densenet121-a639ec97.pth'
    
    @property
    def path(self):
        return r'./image_ai/classification/model/storage/' + self.value
    