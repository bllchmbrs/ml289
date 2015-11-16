import numpy as np


def t(z):
    return np.tanh(z)


def dt(z):
    return 1 - np.power(np.tanh(z), 2)


def g(z):
    return 1 / (1 + np.exp(-1 * z))


def dg(z):
    return g(z) * (1 - g(z))


class NeuralNet:
    def __init__(self, num_input_nuerons, num_hidden_nuerons,
                 num_output_nuerons,
                 random_seed=1):

        print("Initiated")
        pass

    def cache(self, name):
        "Save the current weights to disk"
        pass

    def load_cache(self, name):
        pass

    def train(self, inputs, labels, params):
        "Train our Network Using the Inputs and Labels"
        pass

    def predict(self, inputs):
        pass
