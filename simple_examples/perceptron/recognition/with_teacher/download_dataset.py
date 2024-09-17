""" Preparing anf loading dataset """

import pandas as pd
import numpy as np
import certifi
import os
from pathlib import Path


def makepath(s: str) -> str:
    cur_path = (Path(__file__).parent / 'data').resolve()
    if not os.path.exists(cur_path):
        os.makedirs(cur_path)
    return str(cur_path / f'{s}.npy')


def download():
    # I couldn't download Iris.csv without these strings (on Mac OS)
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    os.environ["SSL_CERT_FILE"] = certifi.where()


    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    df = pd.read_csv(url, header=None)

    X = df.iloc[0:100, [0, 2]].values
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    with open(makepath('X'), 'wb') as Xo, \
        open(makepath('y'), 'wb') as yo:
        np.save(Xo, X)
        np.save(yo, y)
