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
        self.w1 = np.random.uniform(low=0, high=1, size=input_layer.shape)
        # a set of weights and biases between hidden layer to output layer.
        self.w2 = np.random.uniform(low=0, high=1, size=output_layer.shape)
        # an hidden layer.
        self.layer1 = np.zeros(hidden_size)
        # an output layer size.
        self.output_layer = output_layer
        # an output layer.
        self.output = np.zeros(output_layer.shape)
        # a activation function.
        self.activation, self.activation_derivative = NeuralNetwork.get_activation(activation)

    def feedforward(self):
        """a neural network, predictions are made based on the values in the input nodes and the weights.

        :return:
        """
        # update hidden layer.
        self.layer1 = np.asarray([self.activation(np.sum(subarr)) for subarr in
                                  np.split(np.dot(self.input, self.w1), self.layer1.shape[0])])
        # resize layer1 to output size. (for simple multiply with w2)
        layer1_resize = np.asarray([np.full(self.layer1.shape[0], val)for val in self.layer1])
        # append all subarray to one dimension array.
        layer1_resize = np.append(layer1_resize)
        # update output layer.
        self.output = self.activation(np.dot(layer1_resize, self.w2))

    def back_propagation(self):
        """minimizing the cost of a loss function, using sum of squares error.

        :return:
        """
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        delta_w2 = np.dot(self.layer1.T, (2 * (self.output_layer - self.output) *
                                          self.activation_derivative(self.output)))
        delta_w1 = np.dot(self.input.T, (np.dot(2 * (self.output_layer - self.output) *
                                                self.activation_derivative(self.output),
                                                self.w2.T) * self.activation_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.w1 += delta_w1
        self.w2 += delta_w2

    def change_activation(self, activation):
        """change activation function.

        :param activation: str, a activation function name.
        :return:
        """
        self.activation = NeuralNetwork.get_activation(activation=activation)

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

    for i in range(1500):
        nn.feedforward()
        nn.back_propagation()

    print(nn.output)
