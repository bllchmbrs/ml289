import numpy as np
import pandas as pd
import scipy as sp
import scipy.io
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle


def load_training_data():
    raw_train_images = scipy.io.loadmat(
        "dataset/train.mat")['train_images'].T.reshape(60000, 784)

    raw_train_labels = scipy.io.loadmat(
        "dataset/train.mat")['train_labels'].reshape(60000, )

    raw_train_images, raw_train_labels = shuffle(raw_train_images,
                                                 raw_train_labels)

    tX, XVal, ty, yVal = train_test_split(raw_train_images, raw_train_labels,
                                          test_size=0.22)
    tX = tX / 255
    XVal = XVal / 255

    return {"X": tX, "y": ty, "XVal": XVal, "yVal": yVal}


def load_test_data():
    print(scipy.io.loadmat("dataset/test.mat")['test_images'].T.shape)
    kaggleTest = scipy.io.loadmat("dataset/test.mat")['test_images'].T.reshape(
        10000, 784)
    kaggleTest = normalize(kaggleTest)
    print(kaggleTest.shape)
    return kaggleTest
