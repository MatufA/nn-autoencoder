from NeuralNetwork import NeuralNetwork
import numpy as np


class Autoencoder:
    def __init__(self, image_path, encode_size, activation="sigmoid"):
        """an limited implementation of Autoencoder Neural Network with
        one hidden layer as a half size of a input layer.

        :param image_path: str, an image path.
        :param encode_size: int, an image resize size. (rectangle, x * x)
        """
        self.activation = activation
        self.image_path = image_path
        self.encode_size = encode_size
        self.loss_error = []
        self.input_layer = np.random.uniform(low=0, high=1, size=encode_size * encode_size)
        self.nn = NeuralNetwork(input_layer=self.input_layer, hidden_size=self.input_layer.shape[0]/2,
                                output_layer=self.input_layer, activation=activation)

    def change_activation(self, activation):
        """change activation function.

        :param activation: str, a activation function name.
        :return:
        """
        self.activation = self.nn.change_activation(activation=activation)

    def draw_graph(self):
        """draw a graph of a loss function.

        :return:
        """
        pass

    def fit(self):
        """fit an image to input layer model.

        :return: an input layer.
        """
        pass

    def train(self, epoch=1000):
        """train the model.

        :param epoch: int, an amount of loops for full train model.
        :return:
        """
        pass

    def test(self):
        """test the model.

        :return:
        """
        pass

    @staticmethod
    def image_to_np_array(image_path):
        """Change image from binary to numpy array.

        :param image_path: str, an image input path.
        :return: a numpy array.
        :rtype: numpy array dtype int.
        """
        from matplotlib.image import imread
        return np.asarray(imread(image_path), dtype=np.int)

    @staticmethod
    def split_image_chucks(image_path, chunks_size):
        """split a chucks from image to fit the to input layer of the nn.

        :param chunks_size: int, a size of chunks. (rectangle [x,x]
        :param image_path: str, an image input path.
        :return: a list ot image chunks.
        :type: numpy array.
        """
        img_ndarray = Autoencoder.image_to_np_array(image_path=image_path)
        img = np.split(img_ndarray, img_ndarray.shape[0] / chunks_size)
        chunks = []
        for d in img:
            chunks.extend(np.split(d, d.shape[1] / chunks_size, axis=1))
            # for chunk in chunks:
            #     yield chunk
        return np.asarray(chunks)

    @staticmethod
    def append_image_chucks(chunks):
        """appends a chucks from image to fit the to input layer of the nn.

        :param chunks: numpy array, an image chunks.
        :return:
        """
        chunks_size = chunks.shape[1]
        image = []
        for idx in range(chunks_size):
            image.extend(np.concatenate(([[]], chunks[idx * chunks_size:(idx+1) * chunks_size - 1]), axis=1))
        return np.asarray(np.concatenate(([[], image])), dtype=int)

    @staticmethod
    def show_image_from_np(image_np):
        """show image from numpy array.

        :param image_np: numpy array, an image as numpy array..
        :return:
        """
        from matplotlib import pyplot as plt
        plt.imshow(image_np, interpolation='nearest').show()


if __name__ == '__main__':
    data = Autoencoder.split_image_chucks(image_path="E:/source/nn-autoencoder/data/in/Photo_of_Lena_in_ppm.jpg",
                                          chunks_size=16)
    np_image = Autoencoder.append_image_chucks(chunks=data)
    print(np_image.shape)
    # Autoencoder.show_image_from_np(image_np=data)
