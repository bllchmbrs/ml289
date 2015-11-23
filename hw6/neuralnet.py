import numpy as np
import pandas as pd
import datetime


def shape_check(checks):
    for k, v in checks.items():
        try:
            assert k == v
        except AssertionError:
            print("ERROR")
            print(k, v.shape)
            raise AssertionError


class Layer:
    def __init__(self, shape, activation_func, activation_prime, seed=10):
        self.layer_type = "Mid"
        np.random.seed(seed)
        self.rows, self.cols = shape
        self.z = np.zeros(self.cols, dtype=np.float64)
        self.y = np.zeros(self.cols, dtype=np.float64)
        self.b = np.ones(self.cols, dtype=np.float64)
        self.w = np.random.normal(0, 0.01, shape)

        self.a_func = activation_func
        self.a_prime = activation_prime

        self.delta = np.zeros(self.cols)
        self.update = np.zeros(shape)

    def debug_shapes(self):
        print(self.layer_type)
        print("y", self.y.shape)
        print("z", self.z.shape)
        print("w", self.w.shape)
        print("delta", self.delta.shape)
        print("update", self.update.shape)

    def check_shapes(self):
        cxr = (self.cols, )
        shape_check({
            cxr: self.z.shape,
            cxr: self.y.shape,
            cxr: self.z.shape,
            (self.rows, self.cols): self.w.shape
        })

    def forward_pass(self, data):
        data.dot(self.w, out=self.z)
        self.z += self.b
        self.y = self.a_func(self.z)
        return self.y

    def calc_delta(self, previous_delta):
        np.multiply(previous_delta, self.a_prime(self.z), out=self.delta)
        return self.delta

    def update_weights(self, previous_y, eta):
        np.outer(previous_y, self.delta, out=self.update)
        self.w -= (eta * self.update)
        self.b -= (eta * self.delta)

    def transform(self, data):
        return self.a_func(data.dot(self.w) + self.b)


class OutputLayer(Layer):
    def __init__(self, shape, activation_func, activation_prime, seed=10):
        super().__init__(shape, activation_func, activation_prime, seed)
        self.layer_type = "Out"


class NeuralNet:
    def __init__(self, hidden_layer, output_layer, cost_func, cost_prime,
                 seed=10):

        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

        self.transformations = [self.hidden_layer.transform,
                                self.output_layer.transform]
        self.cost_func = cost_func
        self.cost_prime = cost_prime

        self.val_scores = []
        self.train_params = {}  # filled in when we train

    def check_shapes(self):
        self.hidden_layer.check_shapes()
        self.output_layer.check_shapes()

    def train(self, inputs, labels, **kwargs):
        "Train our Network Using the Inputs and Labels"
        input_dim = len(inputs[0])
        output_dim = len(labels[0])

        val_score = False
        XVal, yVal = kwargs.get("XVal", None), kwargs.get("yVal", None)

        self.train_params = kwargs
        del self.train_params['XVal']
        del self.train_params['yVal']

        num_iters = kwargs.get("num_iters", 1000)
        eta = kwargs.get("eta", 0.01)
        score_every = kwargs.get("score_every", num_iters / 10)

        if XVal != None:
            val_score = True
        start = datetime.datetime.now()
        self.train_params['start'] = start

        for _ in range(num_iters):
            ex_index = np.random.choice(len(inputs), 1)
            y_0 = inputs[ex_index].reshape(input_dim, )
            y = labels[ex_index].reshape(output_dim, )

            y_1 = self.hidden_layer.forward_pass(y_0)
            y_hat = self.output_layer.forward_pass(y_1)

            self.check_shapes()
            delta_out = self.cost_prime(y, y_hat)
            delta2 = self.output_layer.calc_delta(delta_out)
            delta1 = self.hidden_layer.calc_delta(
                delta2.dot(self.output_layer.w.T))
            self.check_shapes()

            self.output_layer.update_weights(self.hidden_layer.y, eta)
            self.hidden_layer.update_weights(y_0, eta)

            if val_score and _ % score_every == 0:
                v_score = self.score(XVal, yVal)
                self.val_scores.append((v_score, v_score / len(XVal), _))

        end = datetime.datetime.now()
        self.train_params['end'] = end
        print("Total Train Time %.2f Seconds" % (end - start).total_seconds())

    def predict(self, inputs):
        temp = inputs
        for transformation in self.transformations:
            temp = transformation(temp)
        return temp.argmax(axis=1)

    def score(self, inputs, labels):
        preds = self.predict(inputs)
        return sum(preds == labels.argmax(axis=1))

    def resulting_scores(self):
        scoredf = pd.DataFrame(self.val_scores)
        scoredf.columns = ['val', 'val_accuracy', 'iter']
        return scoredf
