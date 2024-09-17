"""" Save and load dataset """

from pathlib import Path
import os

from keras.api.models import load_model

from cnn import datasets, net_v2


def save():
    my_datasets = datasets()
    model = net_v2(*my_datasets)
    model.save(_file())
    

def load():
    if not os.path.exists(_file()):
        save()
    model = load_model(_file())
    return model


def _file() -> Path:
    cur_path = (Path(__file__).parent / 'data').resolve()
    if not os.path.exists(cur_path):
        os.makedirs(cur_path)
    return cur_path / 'model.h5'
