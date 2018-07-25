import numpy as np
from sklearn.utils import shuffle as sk_shuffle

class NeuralNetwork:

    # assumes only a single hidden layer, each of specified dimensions
    def __init__(self, num_in, num_hidden, num_out, lr=0.001, epoch=3000):
        self.input_size = num_in
        self.hidden_size = num_hidden
        self.output_size = num_out
        self.lr = lr
        self.epoch = epoch

        # initialize weights and biases
        self.w_h = np.random.uniform(low=-1, high=1, size=(self.input_size, self.hidden_size))
        self.b_h = np.random.uniform(low=-1, high=1, size=(1, self.hidden_size))

        self.w_out = np.random.uniform(low=-1, high=1, size=(self.hidden_size, self.output_size))
        self.b_out = np.random.uniform(low=-1, high=1, size=(1, self.output_size))

        self.hidden_input = 0
        self.hidden_activations = 0
        self.output_input = 0

    # forward propagation, returns array of output values in range [0, 1], assumes x is a matrix of input vectors
    def forward_prop(self, x):
        self.hidden_input = np.dot(x, self.w_h) + self.b_h
        self.hidden_activations = self.sigmoid(self.hidden_input)

        self.output_input = np.dot(self.hidden_activations, self.w_out) + self.b_out
        output_activations = self.sigmoid(self.output_input)

        return output_activations

    # back propagation through standard gradient descent
    def back_prop(self, x, y, output):

        # using sum of squared errors loss function
        E = y - output

        delta_out = E * self.deriv_sigmoid(output)

        error_h = delta_out.dot(self.w_out.T)
        delta_h = error_h * self.deriv_sigmoid(self.hidden_activations)

        self.w_out += self.hidden_activations.T.dot(delta_out) * self.lr
        self.b_out += np.sum(delta_out, axis=0, keepdims=True) * self.lr

        self.w_h += x.T.dot(delta_h) * self.lr
        self.b_h += np.sum(delta_h, axis=0, keepdims=True) * self.lr

    # stores weights and biases for future use
    def store_weights_biases(self):

        np.savetxt('wh.txt', self.w_h.view(float))
        np.savetxt('wo.txt', self.w_out.view(float))
        np.savetxt('bh.txt', self.b_h.view(float))
        np.savetxt('bo.txt', self.b_out.view(float))

    # loads weighs and biases from respective text files
    def load_weights_biases(self, wh=np.loadtxt('wh.txt'),
                            wo=np.loadtxt('wo.txt'),
                            bh=np.loadtxt('bh.txt'),
                            bo=np.loadtxt('bo.txt')):
        self.w_h = wh
        self.w_out = wo
        self.b_h = bh
        self.b_out = bo

    # activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of activation function
    def deriv_sigmoid(self, x):
        return x * (1 - x)

    # fit the model with training data using mini-batch gradient descent via the back_prop() method
    def train(self, x, y, minibatch_size):
        for i in range(self.epoch):

            x, y = sk_shuffle(x, y)

            for j in range(0, x.shape[0], minibatch_size):
                output = self.forward_prop(x)
                self.back_prop(x, y, output)

    # returns prediction using forward propogation
    def predict(self, tests):
        output = self.forward_prop(tests)
        return np.argmax(output)
