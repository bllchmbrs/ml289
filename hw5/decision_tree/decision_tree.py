import numpy as np
import pandas as pd
from scipy.stats import entropy, itemfreq
from numpy.random import choice


def quick_stats(data):
    return "Length: %i" % (len(data))


class DecisionTree:
    def __init__(self, params):
        self.node = None
        self.max_depth = params['max_depth']
        self.min_points = params['min_points']

    def pretrain_check(self, train_data, train_labels):
        if self.max_depth == 0:
            return False
        if len(train_labels) < self.min_points:
            return False
        return True

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Decision Tree\n Max depth: %i \n Node: %s" % (self.max_depth,
                                                              str(self.node))

    def entropy_from(self, labels):
        probs = itemfreq(labels)[:, 1] / len(labels)
        return entropy(probs)

    def impurity(self, left_labels, right_labels):
        total = len(left_labels) + len(right_labels)
        left_entropy = self.entropy_from(left_labels)
        right_entropy = self.entropy_from(right_labels)

        return np.sum([left_entropy * len(left_labels) / total,
                       right_entropy * len(right_labels) / total])

    def segmentor(self, data, labels):
        impurities = []
        for feature in choice(len(data[0,:]), 5):
            # print("Using feature number: %i" % feature)
            for unique_val in np.unique(data[:, feature]):
                pc1 = labels[np.where(data[:, feature] <= unique_val)]
                # maybe just less than?
                pc2 = labels[np.where(data[:, feature] > unique_val)]
                impurities.append({
                    "feature": feature,
                    "split": unique_val,
                    "impurity": self.impurity(pc1, pc2)
                })
        best = min(impurities, key=lambda x: x['impurity'])
        print(best)
        return best['feature'], best['split']

    def train(self, train_data, train_labels):  # could add random seed
        if not self.pretrain_check(train_data, train_labels):
            return  # don't pass the check, don't train
        print(quick_stats(train_data))

        shuff = choice(len(train_labels), len(train_labels))
        train_data = train_data[shuff]
        train_labels = train_labels[shuff]

        feature, split = self.segmentor(train_data, train_labels)
        self.node = Node(feature, split)
        ldata, rdata, llabels, rlabels = self.node.apply(train_data,
                                                         train_labels)

        self.node.left = DecisionTree(
            {"max_depth": self.max_depth - 1,
             "min_points": self.min_points})
        self.node.right = DecisionTree(
            {"max_depth": self.max_depth - 1,
             "min_points": self.min_points})
        print("training left node - depth: %i" % (4 - self.max_depth))
        self.node.left.train(ldata, llabels)
        print("training right node - depth: %i" % (4 - self.max_depth))
        self.node.right.train(rdata, rlabels)

    def predict(self, test_data):
        pass


class Node:
    def __init__(self, feature, split_rule, label=None):
        self.parent = None
        self.feature = feature  # feature number
        self.split_rule = split_rule
        self.label = label
        self.left = None
        self.right = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Feature: %i, Rule: %.3f" % (self.feature, self.split_rule)

    def apply(self, data, labels=None):
        left = np.where(data[:, self.feature] <= self.split_rule)
        right = np.where(data[:, self.feature] > self.split_rule)

        if type(labels) != type(None):
            return data[left], data[right], labels[left], labels[right]

        return data[left], data[right]
