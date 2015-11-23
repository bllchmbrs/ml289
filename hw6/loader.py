import numpy as np
import pandas as pd
import scipy as sp
import scipy.io
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.datasets import make_classification


def load_training_data():
    raw_train_images = scipy.io.loadmat(
        "dataset/train.mat")['train_images'].T.reshape(60000, 784) / 255.0
    raw_train_labels = scipy.io.loadmat(
        "dataset/train.mat")['train_labels'].reshape(60000, )
    tX, XVal, ty, yVal = train_test_split(raw_train_images, raw_train_labels,
                                          test_size=0.25,
                                          random_state=10)
    tX = tX
    XVal = XVal

    return {"X": tX, "y": ty, "XVal": XVal, "yVal": yVal}


def load_test_data():
    return scipy.io.loadmat("dataset/test.mat")['test_images'].T.reshape(
        10000, 784) / 255.0


def fake_data():
    X, y = make_classification(1100,
                               n_classes=2,
                               n_features=2,
                               n_informative=2,
                               n_redundant=0,
                               n_repeated=0)
    pd.DataFrame({"x1": X[:, 0],
                  "x2": X[:, 1],
                  "y": y}).plot(kind='scatter',
                                x='x1',
                                y='x2',
                                c='y')
    y = pd.get_dummies(y).values
    return X, y
