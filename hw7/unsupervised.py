import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib


class KMeans:
    def __init__(self, k=5, num_iters=10, seed=10):
        self.k = k
        self.num_iters = num_iters
        self.losses = []
        self.seed = seed
        self.means = None
        self.start = None  # keep track of where we started

    def fit(self, X, y=None):
        n_examples, n_features = X.shape
        np.random.seed(self.seed)
        start = X[np.random.choice(len(X), self.k, False)]
        start = np.abs(np.random.normal(X.mean(axis=0).mean(),
                                        X.mean(axis=0).std(),
                                        size=(self.k, n_features)))
        self.start = start
        self.means = start
        for it in range(self.num_iters):
            assignments = self.predict(X)
            for mean in range(self.k):
                self.means[mean] = X[assignments == mean].mean(axis=0)
        self.mean_stats(X)
        return self

    def loss(self, pred_values):
        loss = []
        assignments = pred_values.argmin(axis=1)
        for mean in range(self.k):
            n_assign = pred_values[assignments == mean].sum()
            loss.append(n_assign / len(assignments))
        self.losses.append(loss)

    def mean_stats(self, X):
        preds = self.predict(X)
        pd.DataFrame(self.losses).plot()
        pd.DataFrame(self.losses).sum(axis=1).plot()
        for mean in range(self.k):
            print("Number in Mean Number %i %i" %
                  (mean, len(X[preds == mean])))

    def plot_means(self, figsize=(10, 8)):
        fig = plt.figure(figsize=figsize)
        gs = matplotlib.gridspec.GridSpec(ceil(self.k / 2), 2,
                                          width_ratios=[1, 1])
        for mean, ax in zip(range(self.k), gs):
            sp = plt.subplot(ax)
            sp.imshow(self.means[mean].reshape((28, 28)))
            sp.set_title("Mean Number: %i" % mean)
            sp.set_axis_off()

    def predict(self, X, y=None):
        pred_values = cdist(X, self.means, 'euclidean')
        self.loss(pred_values)
        return pred_values.argmin(axis=1)
