import numpy as np


class NeuralNetwork:
    def __init__(self, input_layer, hidden_size, output_layer, activation='sigmoid'):
        """
        :param input_layer: list, an input vector.
        :param hidden_size: int, an hidden layer size.
        :param output_layer: int, an output layer.
        :param activation: str, the activation function to be used. ("sigmoid", "relu or "tanh")
        """
        # an input layer.
        self.input = np.asarray(input_layer)
        # a set of weights and biases between input layer to hidden layer.
        self.w1 = np.random.uniform(low=0, high=1, size=(input_layer.shape[0], hidden_size)).astype(dtype='float64')
        # a set of weights and biases between hidden layer to output layer.
        self.w2 = np.random.uniform(low=0, high=1, size=(hidden_size, output_layer.shape[0])).astype(dtype='float64')
        # an hidden layer.
        self.layer1 = np.zeros(hidden_size)
        # an output layer size.
        self.output_layer = output_layer
        # an output layer.
        self.output = np.zeros(output_layer.shape)
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
        self.layer1 = self.activation(np.dot(self.input, self.w1))
        # update output layer.
        self.output = self.activation(np.dot(self.layer1, self.w2))
        return self.output

    def back_propagation(self):
        """minimizing the cost of a loss function, using sum of squares error.

        :return:
        """
        # calculate the error of output layer.
        output_error = self.output_layer - self.output
        delta_output = 2 * output_error * self.activation_derivative(self.output)
        # calculate the error of hidden layer.
        error_layer1 = np.dot(delta_output, self.w2.T)
        # find delta of wight 2
        delta_layer1 = error_layer1 * self.activation_derivative(self.layer1)
        # update the weights with the derivative of the loss function
        self.w1 += np.sum([self.input * cell for cell in delta_layer1])
        self.w2 += np.sum([delta_output * cell for cell in self.layer1])
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
        self.input = value

    def set_output(self, value):
        """set output layer.

        :param value: value: numpy array, an output layer.
        """
        self.output_layer = value

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


if __name__ == "__main__":
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    y = np.array([[0], [1], [1], [0]])
    nn = NeuralNetwork(input_layer=X, hidden_size=4, output_layer=y, activation="sigmoid")
    print(nn.layer1.shape)

    for i in range(1500):
        nn.feedforward()
        nn.back_propagation()

    print(nn.output)
