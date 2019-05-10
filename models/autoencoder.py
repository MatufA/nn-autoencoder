from NeuralNetwork import NeuralNetwork
from config import conf_logger
from datetime import datetime
import numpy as np
import logging


class Autoencoder:
    def __init__(self, input_layer_len, hidden_layer_len, output_layer_len, activation="sigmoid"):
        """an limited implementation of Autoencoder Neural Network with
        one hidden layer as a half size of a input layer.

        :param input_layer_len: numpy array, an input data for the input layer.
        :param output_layer_len: numpy array, an output data for the input layer.
        :param activation: str, a activation function name. (default sigmoid)
        """
        self.activation = activation
        self.input_layer = np.random.uniform(low=0, high=1, size=input_layer_len)
        self.output_layer = np.random.uniform(low=0, high=1, size=output_layer_len)
        self.nn = NeuralNetwork(input_layer=self.input_layer, hidden_size=hidden_layer_len,
                                excpected=self.output_layer, activation=activation)
        self.loss_error = []

    def change_activation(self, activation):
        """change activation function.

        :param activation: str, a activation function name.
        :return:
        """
        self.activation = self.nn.change_activation(activation=activation)

    def draw_graph(self, folder_path):
        """draw a graph of a loss function to screen and to file.

        :param folder_path: str, a path to folder for draw a graph.
        :return:
        """
        import matplotlib.pyplot as plt

        # Data for plotting
        points = np.asarray(self.loss_error) / 1e-9
        epoch = np.arange(0, len(self.loss_error))

        fig, ax = plt.subplots()
        ax.plot(epoch, points)

        ax.set(xlabel='epoch (num)', ylabel='loss function (sum)',
               title='Loss function over epoch.')
        ax.grid()

        fig.savefig("{}/{}.png".format(folder_path, datetime.now().strftime("%Y-%m-%d_%H-%M")))
        plt.show()

    @staticmethod
    def fit_transform(data_to_fit):
        """fit an image to input layer model.

        :param data_to_fit: numpy array, a multi dimensions array.
        :return: an one dimension array.
        :rtype: numpy array.
        """
        return np.asarray([np.ravel(im) for im in data_to_fit], dtype='float64')

    def train(self, train_data, alpha, epoch=1000):
        """train the model.

        :param train_data: numpy array, a data to train the model.
        :param alpha: a learning rate. (0.1 => 10%)
        :param epoch: int, an amount of loops for full train model.
        :return:
        """
        for sample in train_data:
            # set input layer of nn.
            self.nn.set_input(value=sample)
            # set output layer of nn.
            self.nn.set_output(value=sample)
            # train epoch
            for i in range(epoch):
                # feedforward.
                self.nn.feedforward()
                # back propagation.
                self.nn.back_propagation(alpha=alpha)
                # print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
                self.loss_error.append(self.nn.loss_error)

    def predict(self, predict_val):
        """predict the output the model.

        :return:
        """
        # root folder path.
        root = path.dirname(path.dirname(__file__))
        # set input layer of nn.
        self.nn.set_input(value=predict_val)
        # set output layer of nn.
        self.nn.set_output(value=predict_val)
        # load weight.
        self.nn.load_weights("{}/data/out".format(root))
        # feedforward.
        out_img = self.nn.feedforward()
        # draw a output
        Autoencoder.save_image_from_np(image_np=out_img,
                                       to_path="{0}/data/out/predicted_{1}.png"
                                       .format(root, datetime.now().strftime("%Y-%m-%d_%H-%M")))

    @staticmethod
    def train_test_split(data, train_split):
        """split data randomly.

        :param data: a data to split.
        :param train_split: float, a size of train sets. (0 < x < 1)
        :return: a array of train data.
        """
        train_test = []
        if 0 < train_split < 1:
            index = np.random.choice(data.shape[0], int(data.shape[0]*train_split), replace=False)
            for idx in index:
                train_test.append(data[idx])
        return np.asarray(train_test, dtype='float64')

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
    def split_image_chunks(image_path, chunks_size):
        """split a chucks from image to fit the to input layer of the nn.

        :param chunks_size: int, a size of chunks. (rectangle [x,x]
        :param image_path: str, an image input path.
        :return: a list ot image chunks.
        :type: numpy array.
        """
        img_ndarray = Autoencoder.image_to_np_array(image_path=image_path)
        print(img_ndarray.shape)
        img = np.asarray(np.split(img_ndarray, img_ndarray.shape[0] / chunks_size))
        print(img.shape)
        chunks = []
        for d in img:
            chunks.extend(np.split(d, d.shape[1] / chunks_size, axis=1))
        return np.asarray(chunks)

    @staticmethod
    def append_image_chunks(np_arr, chunks_size):
        """appends a chucks from image to fit the to input layer of the nn.

        :param chunks_size: int, a size of chunks. (rectangle [x,x])
        :param np_arr: str, an numpy array image chunks.
        :return: a list ot image chunks.
        :type: numpy array.
        """
        img = []
        for pixel in np_arr:
            img.append(np.append(pixel, axis=1))
        return

    @staticmethod
    def save_image_from_np(image_np, to_path):
        """save image from numpy array.

        :param to_path: str, a path to save an image.
        :param image_np: numpy array, an image as numpy array..
        :return:
        """
        from matplotlib.image import imsave
        imsave(to_path, image_np)

    @staticmethod
    def show_image_from_np(image_np):
        """show image from numpy array.

        :param image_np: numpy array, an image as numpy array..
        :return:
        """
        from matplotlib import pyplot as plt
        plt.imshow(image_np, interpolation='nearest').show()


if __name__ == '__main__':
    from os import path, makedirs
    from shutil import rmtree

    # root folder path.
    root_folder = path.dirname(path.dirname(__file__))

    # cofigure logger
    conf_logger(log_path="{}/log".format(root_folder), file_name="autoencoder")
    logger = logging.getLogger("autoencoder")
    logger.info("starting Autoencoder model.")

    # chonks size
    chunk_size = 16
    # split the image to small
    logger.info("Splitting image.")
    data = Autoencoder.split_image_chunks(image_path="{}/data/in/Photo_of_Lena_in_ppm.jpg".format(root_folder),
                                          chunks_size=chunk_size)
    print(data.shape)

    logger.info("creating train folder with splitting images.")
    # train folder path.
    train_folder = "{}/data/train".format(root_folder)
    # check if train folder exists.
    if path.exists(train_folder):
        logger.debug("removing train folder.")
        # remove train folder.
        rmtree(train_folder, ignore_errors=True)
        logger.debug("creating new train folder.")
        # make new train folder.
        makedirs(train_folder)
    else:
        logger.debug("creating train folder.")
        # create train folder.
        makedirs(train_folder)

    logger.debug("saving image chunks to train folder.")
    # save images chunks to train folder.
    for idx, im in enumerate(data):
        Autoencoder.save_image_from_np(image_np=im,
                                       to_path="{}/data/train/{:04d}.jpg".format(root_folder, idx))

    # train split len
    train_split = 0.75
    # choose randomly train set.
    logger.info("choosing randomly train set of size {}".format(int(data.shape[0] * train_split)))
    train = Autoencoder.train_test_split(data=data, train_split=train_split) / 255

    # fit data to input of the model.
    logger.info("fitting the chunks to input of the model.")
    flat_train = Autoencoder.fit_transform(data_to_fit=train)

    # create a Autoencoder object.
    logger.info("initialing autoencoder Neureal Network.")
    input_output_len = chunk_size * chunk_size
    autoencoder = Autoencoder(input_layer_len=input_output_len,
                              hidden_layer_len=int(input_output_len / 2),
                              output_layer_len=input_output_len)
    logger.info("training the model.")
    # train the data.
    autoencoder.train(train_data=flat_train, alpha=0.2, epoch=100)
    autoencoder.nn.save_weights("{}/data/out".format(root_folder))
    logger.info("drawing graph.")
    # draw graph.
    autoencoder.draw_graph(folder_path="{}/data/out".format(root_folder))
    # logger.info("loading Lena image.")
    # # load image to predict.
    # np_image = Autoencoder.image_to_np_array(image_path="{}/data/in/Photo_of_Lena_in_ppm.jpg".format(root_folder))
    # logger.info("predicting Lena image.")
    # # predict
    # autoencoder.predict(predict_val=np_image)
