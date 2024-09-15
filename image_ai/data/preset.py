""" Test files """

import os
from enum import Enum


_IMG_STORAGE_PATH = './image_ai/data/storage/img'
_IMG_OUTPUT_PATH = './image_ai/data/output/img'
_VIDEO_STORAGE_PATH = './image_ai/data/storage/video'
_VIDEO_OUTPUT_PATH = './image_ai/data/output/video'


def output_path(model_type, out, value):
    out_path = f'{out}/{model_type}/'
    if not os.path.exists(out_path):
        out_path = os.path.abspath(out_path) 
        os.makedirs(out_path)
    return rf'{out_path}/{value}'


class PictureStorage(Enum):
    """ Filenames of pictures """
    PLANE = '1.jpg'
    CAR = '2.jpg'
    DOG = '3.jpg'
    STATIONERY = '4.jpg'
    MUSHROOMS = '5.jpg'
    PERSON = '6.jpg'
    ANIME = '7.jpg'

    @property
    def path(self):
        return f'{_IMG_STORAGE_PATH}/{self.value}'

    def output_path(self, model_type):
        return output_path(model_type, _IMG_OUTPUT_PATH, self.value)
    

class VideoStorage(Enum):
    """ Filenames of videos """
    TRAFFIC = '1.mp4'
    BEACH = '2.mp4'
    
    @property
    def path(self):
        return f'{_VIDEO_STORAGE_PATH}/{self.value}'

    def output_path(self, model_type):
        return output_path(model_type, _VIDEO_OUTPUT_PATH, self.name.lower())