""" Test files """

import os
from enum import Enum


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
        return rf'./image_ai/data/storage/{self.value}'
    
    def output_path(self, model_type):
        out_path = rf'./image_ai/data/output/{model_type}/'
        if not os.path.exists(out_path):
            out_path = os.path.abspath(out_path) 
            os.makedirs(out_path)
        return rf'{out_path}/{self.value}'
