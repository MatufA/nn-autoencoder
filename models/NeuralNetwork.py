import numpy as np


class NeuralNetwork:
    def __init__(self, input_layer, hidden_size, expected, activation='sigmoid'):
        """
        :param input_layer: numpy array, an input vector.
        :param hidden_size: int, an hidden layer size.
        :param expected: numpy array, an expected output.
        :param activation: str, the activation function to be used. ("sigmoid", "relu or "tanh")
        """
        # an input layer.
        self.input = np.asarray(input_layer)
        # a set of weights and biases between input layer to hidden layer.
        self.w1 = np.random.uniform(low=0, high=1, size=(input_layer.shape[0], hidden_size)).astype(dtype='float64')
        # a set of weights and biases between hidden layer to output layer.
        self.w2 = np.random.uniform(low=0, high=1, size=(hidden_size, expected.shape[0])).astype(dtype='float64')
        # an hidden layer.
        self.layer1 = np.zeros(hidden_size)
        # an hidden layer after activation function.
        self.layer1_activation = np.zeros(hidden_size)
        # an output layer size.
        self.expected = expected
        # an output layer.
        self.output = np.zeros(expected.shape)
        # an output layer after activation function.
        self.output_activation = np.zeros(expected.shape)
        # a activation function.
        self.activation, self.activation_derivative = NeuralNetwork.get_activation(activation)
        # a loss function list of sum.
        self.loss_error = 0

    def feedforward(self):
        """a neural network, predictions are made based on the values in the input nodes and the weights.

        :return: an output vector.
        :rtype: numpy array.
        """
        # update hidden layer.
        self.layer1 = np.dot(self.input, self.w1) / self.layer1.shape[0]
        # update hidden layer after activation function.
        self.layer1_activation = self.activation(self.layer1)
        # update output layer.
        self.output = np.dot(self.layer1_activation, self.w2) / self.output.shape[0]
        self.output_activation = self.activation(self.output)
        return self.output

    def back_propagation(self, alpha):
        """minimizing the cost of a loss function, using sum of squares error.

        :param alpha: a learning rate. (0.1 => 10%)
        :return:
        """
        # calculate the error of output layer.
        output_error = self.expected - self.output
        delta_output = output_error * self.activation_derivative(self.output_activation)

        # find error of hidden layer.
        error_layer1 = np.dot(delta_output, self.w2.T)
        # find delta of weight 2
        delta_layer1 = error_layer1 * self.activation_derivative(self.layer1_activation)

        # transpose the delta output vector.
        delta_output = delta_output[np.newaxis]

        # update w2 weight.
        self.layer1_activation = self.layer1_activation.reshape((self.layer1_activation.shape[0], 1))
        self.w2 += np.dot(np.array(self.layer1_activation), delta_output) * alpha
        # update w1 weight.
        self.w1 += np.dot(self.input[np.newaxis].T, delta_layer1[np.newaxis]) * alpha

        # store mean sum squared loss.
        self.loss_error = np.mean(np.square(output_error))

    def change_activation(self, activation):
        """change activation function.

        :param activation: str, a activation function name.
        :return:
        """
        self.activation = NeuralNetwork.get_activation(activation=activation)

    def set_input(self, value):
        """set input layer.

        :param value: numpy array, an input layer.
        """
        self.input = np.asarray(value, dtype='float64')

    def set_expected(self, value):
        """set output layer.

        :param value: value: numpy array, an output layer.
        """
        self.expected = np.asarray(value, dtype='float64')

    def save_weights(self, path):
        """save weights in text file.

        :param path: str, a path to folder location.
        :return:
        """
        np.savetxt(fname="{}/w1.txt".format(path), X=self.w1, fmt="%f")
        np.savetxt(fname="{}/w2.txt".format(path), X=self.w2, fmt="%f")

    def load_weights(self, path):
        """load weights from text file.

        :param path: str, a path to folder location.
        :return:
        """
        self.w1 = np.loadtxt(fname="{}/w1.txt".format(path))
        self.w2 = np.loadtxt(fname="{}/w2.txt".format(path))

    @staticmethod
    def get_activation(activation):
        """get activation function.

        :param activation: str, a activation function name.
        :return: a activation function
        :rtype: function
        """
        if activation == 'sigmoid':
            from activation import sigmoid, sigmoid_derivative
            return sigmoid, sigmoid_derivative
        elif activation == 'relu':
            from activation import relu, relu_derivative
            return relu, relu_derivative
        elif activation == 'tanh':
            from activation import tanh, tanh_derivative
            return tanh, tanh_derivative

