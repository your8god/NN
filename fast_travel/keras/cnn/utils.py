from keras.api.models import load_model

from cnn import datasets, net_v2


_FILE = 'model.h5'


def save():
    datasets = datasets()
    model = net_v2(*datasets)
    model.save(_FILE)
    

def load():
    model = load_model(_FILE)
    return model
