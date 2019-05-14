#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
                                expected=self.output_layer, activation=activation)
        self.loss_error = []

    def change_activation(self, activation):
        """change activation function.

        :param activation: str, a activation function name.
        :return:
        """
        self.activation = self.nn.change_activation(activation=activation)

    def train(self, train_data, alpha, epoch=1000):
        """train the model.

        :param train_data: numpy array, a data to train the model.
        :param alpha: a learning rate. (0.1 => 10%)
        :param epoch: int, an amount of loops for full train model.
        :return:
        """
        folder = path.dirname(path.dirname(__file__))
        # train epoch
        for i in range(epoch):
            for sample in train_data:
                # set input layer of nn.
                self.nn.set_input(value=sample)
                # set output layer of nn.
                self.nn.set_expected(value=sample)
                # feedforward.
                self.nn.feedforward()
                # back propagation.
                self.nn.back_propagation(alpha=alpha)
            self.loss_error.append(self.nn.loss_error)
            print('epoch={epoch}, l_rate={l_rate:.3f}, loss={loss:.5f}'.format(epoch=i,
                                                                               l_rate=alpha,
                                                                               loss=self.nn.loss_error))
        # save last weights.
        autoencoder.nn.save_weights("{}/data/out".format(folder))

    def predict(self, predict_val):
        """predict the output the model.

        :param predict_val: numpy array, input to predict value.
        :return: an autoencoder feed-forward output.
        :rtype: numpy array.
        """
        # root folder path.
        root = path.dirname(path.dirname(__file__))
        out_img = []
        # load weight.
        self.nn.load_weights("{}/data/out".format(root))
        for sample in predict_val:
            # set input layer of nn.
            self.nn.set_input(value=sample)
            # set output layer of nn.
            self.nn.set_expected(value=sample)
            # feedforward.
            out_img.append(self.nn.feedforward())
        return np.array(out_img)

    def draw_graph(self, folder_path, alpha):
        """draw a graph of a loss function to screen and to file.

        :param alpha: float, a learning rate.
        :param folder_path: str, a path to folder for draw a graph.
        :return:
        """
        import matplotlib.pyplot as plt

        # Data for plotting
        points = np.asarray(self.loss_error)
        epoch = np.arange(0, len(self.loss_error))

        fig, ax = plt.subplots()
        ax.plot(epoch, points)

        ax.set(xlabel='epoch (num)', ylabel='loss function (sum)',
               title='Loss function over epoch (lerning rate= {}).'.format(alpha))
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

    @staticmethod
    def reverse_transform(data_to_fit, size):
        """reverse fit transform.

        :param data_to_fit: numpy array, a multi dimensions array.
        :return: an one dimension array.
        :rtype: numpy array.
        """
        revers = [im.reshape((size, size)) for im in data_to_fit]
        return np.asarray(revers, dtype='float64')

    @staticmethod
    def train_test_split(data, train_split):
        """split data randomly.

        :param data: a data to split.
        :param train_split: float, a size of train sets. (0 < x < 1)
        :return: a array of train data.
        """
        train_test = []
        if 0 < train_split < 1:
            index = np.random.choice(data.shape[0], int(data.shape[0] * train_split), replace=False)
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
        img = imread(fname=image_path)
        if len(img.shape) > 2:
            img = img[:, :, 0]
        return np.array(img, dtype='uint8')

    @staticmethod
    def split_row_pixels(row, chunk_size):
        """split row, 1D, to multiply chunks.

        :return:
        """
        return np.array(np.split(np.array(row), chunk_size))

    @staticmethod
    def split_image_chunks(image_path, chunks_size):
        """split a chucks from image to fit the to input layer of the nn.

        :param chunks_size: int, a size of chunks. (rectangle [x,x]
        :param image_path: str, an image input path.
        :return: a list ot image chunks.
        :type: numpy array.
        """
        img_ndarray = Autoencoder.image_to_np_array(image_path=image_path)
        chunks = []
        sub_chunks_size = int(img_ndarray.shape[0] / chunks_size)
        for row_pixels in img_ndarray:
            sub_chunk = np.asarray(Autoencoder.split_row_pixels(row=row_pixels, chunk_size=sub_chunks_size))
            chunks.extend(np.split(sub_chunk, sub_chunks_size / chunks_size))
        return np.array(chunks)

    @staticmethod
    def append_image_chunks(np_arr, original_size):
        """appends a chucks from image to fit the to input layer of the nn.

        :param original_size: int, a size of original image. (rectangle [x,x])
        :param np_arr: str, an numpy array image chunks.
        :return: a list ot image chunks.
        :type: numpy array.
        """
        flat_img = np.append([], np_arr)
        return np.array(np.split(flat_img, original_size))

    @staticmethod
    def save_image_from_np(image_np, to_path):
        """save image from numpy array.

        :param to_path: str, a path to save an image.
        :param image_np: numpy array, an image as numpy array..
        :return:
        """
        from matplotlib.image import imsave
        imsave(to_path, image_np.astype('uint8'))

    @staticmethod
    def show_image_from_np(image_np):
        """show image from numpy array.

        :param image_np: numpy array, an image as numpy array..
        :return:
        """
        from matplotlib import pyplot as plt
        plt.gray()
        plt.figure()
        plt.imshow(image_np)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    from os import path

    # root folder path.
    root_folder = path.dirname(path.dirname(__file__))

    # cofigure logger
    conf_logger(log_path="{}/log".format(root_folder), file_name="autoencoder")
    logger = logging.getLogger("autoencoder")
    logger.info("starting Autoencoder model.")

    # # chunks size
    chunk_size = 16
    # split the image to small chunks
    logger.info("Splitting image.")
    data = Autoencoder.split_image_chunks(image_path="{}/data/in/Photo_of_Lena_in_ppm.jpg".format(root_folder),
                                          chunks_size=chunk_size)

    # fit data to input of the model.
    logger.info("fitting the chunks to input of the model.")
    flat_train = Autoencoder.fit_transform(data_to_fit=data) / 255.0

    # create a Autoencoder object.
    logger.info("initialing autoencoder Neureal Network.")
    input_output_len = chunk_size * chunk_size
    autoencoder = Autoencoder(input_layer_len=input_output_len,
                              hidden_layer_len=int(input_output_len / 2),
                              output_layer_len=input_output_len,
                              activation='tanh')

    # train the data.
    alpha = 0.05
    logger.info("training the model.")
    autoencoder.train(train_data=flat_train, alpha=alpha, epoch=100)

    # draw graph.
    logger.info("drawing graph.")
    autoencoder.draw_graph(folder_path="{}/data/out".format(root_folder), alpha=alpha)

    # load image to predict.
    logger.info("loading image.")
    pred_data = Autoencoder.split_image_chunks(image_path="{}/data/in/cat.gif".format(root_folder),
                                               chunks_size=chunk_size)

    # fit data to input of the model.
    logger.info("fitting the chunks to input of the model.")
    pred_flat = Autoencoder.fit_transform(data_to_fit=pred_data) / 255.0

    # predict
    logger.info("predicting Lena image.")
    pred_out = autoencoder.predict(predict_val=pred_flat) * 255

    # reverse fit transform
    logger.info("changing back to multi dimension.")
    pred_out_mult = Autoencoder.reverse_transform(data_to_fit=pred_out, size=chunk_size)

    # appends chunks to one image.
    logger.info("appending chunks.")
    img_data = Autoencoder.append_image_chunks(np_arr=pred_out_mult, original_size=512).astype('uint8')

    # show image.
    Autoencoder.show_image_from_np(image_np=img_data)

    # save image.
    logger.info("saving image.")
    Autoencoder.save_image_from_np(image_np=img_data,
                                   to_path="{0}/data/out/predicted_{1}.png"
                                   .format(root_folder, datetime.now().strftime("%Y-%m-%d_%H-%M")))
    logger.info("Done.")
