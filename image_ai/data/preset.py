""" Test files """

from enum import Enum


class PictureStorage(Enum):
    """ Filenames of pictures """
    PLANE = '1.jpg'
    CAR = '2.jpg'
    DOG = '3.jpg'
    STATIONERY = '4.jpg'
    MUSHROOMS = '5.jpg'
    WOMAN = '6.jpg'
    
    @property
    def path(self):
        return r'./image_ai/data/storage/' + self.value