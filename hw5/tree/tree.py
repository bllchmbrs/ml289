import numpy as np
import pandas as pd
from scipy.stats import entropy, itemfreq, mode
from numpy.random import choice


def quick_stats(data):
    return "Length: %i" % (len(data))


class RandomForest:
    def __init__(self, params, tree_params):
        self.trees = []
        self.ntrees = params['ntrees']
        self.tree_params = tree_params

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Random Forest. %i Trees" % (self.ntrees)

    def train(self, train_data, train_labels):
        for x in range(self.ntrees):
            bag = choice(len(train_data), len(train_data), True)
            # print("Training tree number %i" % (x))
            tree = DecisionTree(self.tree_params)
            tree.train(train_data[bag], train_labels[bag])
            self.trees.append(tree)
        print("trained all trees")

    def predict(self, data):
        preds = pd.DataFrame(np.array([tree.predict(data)
                                       for tree in self.trees]))
        #print(preds.shape)
        #print(preds)
        return preds.apply(lambda x: x.value_counts().argmax())

    def score(self, test_data, answers):
        return self.predict(test_data) == answers


class DecisionTree:
    def __init__(self, params):
        self.node = None
        self.max_depth = params['max_depth']
        self.min_points = params['min_points']
        self.random_subset = params.get('random_subset', False)
        self.subset_size = params.get('subset_size', 5)

    def pretrain_check(self, train_data, train_labels):
        if self.max_depth == 0:
            return False
        if len(train_labels) < self.min_points:
            return False
        return True

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Decision Tree\n Max depth: %i \n Node: %s\
               \n Random Subset %i\n Subset Size %i" % (self.max_depth,
                                                        str(self.node),
                                                        self.random_subset,
                                                        self.subset_size)

    def entropy_from(self, labels):
        probs = itemfreq(labels)[:, 1] / len(labels)
        return entropy(probs)

    def impurity(self, left_labels, right_labels):
        total = len(left_labels) + len(right_labels)
        left_entropy = self.entropy_from(left_labels)
        right_entropy = self.entropy_from(right_labels)

        return np.sum([left_entropy * len(left_labels) / total,
                       right_entropy * len(right_labels) / total])

    def info_gain(self, cur_labels, left_labels, right_labels):
        return self.entropy_from(cur_labels) - self.impurity(left_labels,
                                                             right_labels)

    def segmentor(self, data, labels, random_subset):
        info_gains = []
        subset = []
        if not random_subset:
            subset = range(len(data[0,:]))
        else:
            subset = choice(len(data[0,:]), self.subset_size, True)
        for feature in subset:
            # print("Using feature number: %i" % feature)
            for unique_val in np.unique(data[:, feature]):
                pc1 = labels[np.where(data[:, feature] <= unique_val)]
                # maybe just less than?
                pc2 = labels[np.where(data[:, feature] > unique_val)]
                info_gains.append({
                    "feature": feature,
                    "split": unique_val,
                    "info_gain": self.info_gain(labels, pc1, pc2)
                })
        best = max(info_gains, key=lambda x: x['info_gain'])
        return best

    def train(self, train_data, train_labels):  # could add random seed
        if not self.pretrain_check(train_data, train_labels):
            return  # don't pass the check, don't train
#        print(quick_stats(train_data))

        shuff = choice(len(train_labels), len(train_labels), False)
        train_data = train_data[shuff]
        train_labels = train_labels[shuff]
        info_gain = self.segmentor(train_data, train_labels,
                                   self.random_subset)
        if info_gain['info_gain'] < 10e-5:
            return
        #print(self)
        self.node = Node(info_gain['feature'], info_gain['split'],
                         mode(train_labels).mode[0])
        ldata, rdata, llabels, rlabels = self.node.apply(train_data,
                                                         train_labels)

        self.node.left = DecisionTree({
            "max_depth": self.max_depth - 1,
            "min_points": self.min_points,
            'random_subset': self.random_subset,
            'subset_size': self.subset_size
        })
        self.node.right = DecisionTree({
            "max_depth": self.max_depth - 1,
            "min_points": self.min_points,
            'random_subset': self.random_subset,
            'subset_size': self.subset_size
        })

        # print("training left node - depth: %i" % (self.max_depth))
        self.node.left.train(ldata, llabels)
        # print("training right node - depth: %i" % (self.max_depth))
        self.node.right.train(rdata, rlabels)

    def predict(self, test_data):
        return np.array([self.node.predict(row.reshape(1, len(row)))
                         for row in test_data])

    def score(self, test_data, answers):
        return self.predict(test_data) == answers


class Node:
    def __init__(self, feature, split_rule, label=None):
        self.feature = feature
        self.split_rule = split_rule
        self.label = label
        self.left = None  # will be of type DecisionTree
        self.right = None  # will be of type DecisionTree

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Feature: %i, Rule: %.3f, label: %i" % (self.feature,
                                                       self.split_rule,
                                                       self.label)

    def apply(self, data, labels):
        left = np.where(data[:, self.feature] <= self.split_rule)
        right = np.where(data[:, self.feature] > self.split_rule)
        return data[left], data[right], labels[left], labels[right]

    def predict(self, row):
        left = np.where(row[:, self.feature] <= self.split_rule)
        right = np.where(row[:, self.feature] > self.split_rule)
        # print(self, "making a prediction")
        # print("Go left: %i, Go Right: %i" % (len(row[left]), len(row[right])))
        # print("Has left: %i, Has Right: %i" %
        #       (self.left.node != None, self.right.node != None))

        if len(row[left]) == 0 and self.right.node != None:
            return self.right.node.predict(row)
        if len(row[right]) == 0 and self.left.node != None:
            return self.left.node.predict(row)
        else:
            return self.label
