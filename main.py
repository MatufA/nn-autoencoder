#!/usr/bin/env python
# -*- coding: utf-8 -*-
from math import exp
from random import seed
from random import random
import numpy as np
from datetime import datetime


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for _ in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for _ in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation / 10))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    folder = path.dirname(path.dirname(__file__))
    mean_error = []
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = row
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        mean_error.append(sum_error)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        # save_weights("{}/data/out".format(folder), ['weights'])
    return mean_error


# Make a prediction with a network
def predict(network, row):
    return forward_propagate(network, row)


def save_weights(path, w1, w2):
    """save weights in text file.

    :param path: str, a path to folder location.
    :return:
    """
    np.savetxt(fname="{}/w2.txt".format(path), X=w2, fmt="%f")
    np.savetxt(fname="{}/w1.txt".format(path), X=w1, fmt="%f")


def fit_transform(data_to_fit):
    """fit an image to input layer model.

    :param data_to_fit: numpy array, a multi dimensions array.
    :return: an one dimension array.
    :rtype: numpy array.
    """
    return np.asarray([np.ravel(im) for im in data_to_fit], dtype='float64')


def reverse_transform(data_to_fit, size):
    """reverse fit transform.

    :param data_to_fit: numpy array, a multi dimensions array.
    :return: an one dimension array.
    :rtype: numpy array.
    """
    revers = [im.reshape((size, size)) for im in data_to_fit]
    return np.asarray(revers, dtype='float64')


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


def image_to_np_array(image_path):
    """Change image from binary to numpy array.

    :param image_path: str, an image input path.
    :return: a numpy array.
    :rtype: numpy array dtype int.
    """
    from matplotlib.image import imread
    return np.asarray(imread(image_path), dtype='uint8')


def split_row_pixels(row, chunk_size):
    """split row, 1D, to multiply chunks.

    :return:
    """
    return np.array(np.split(np.array(row), chunk_size))


def split_image_chunks(image_path, chunks_size):
    """split a chucks from image to fit the to input layer of the nn.

    :param chunks_size: int, a size of chunks. (rectangle [x,x]
    :param image_path: str, an image input path.
    :return: a list ot image chunks.
    :type: numpy array.
    """
    img_ndarray = image_to_np_array(image_path=image_path)
    chunks = []
    sub_chunks_size = int(img_ndarray.shape[0] / chunks_size)
    for row_pixels in img_ndarray:
        sub_chunk = np.asarray(split_row_pixels(row=row_pixels, chunk_size=sub_chunks_size))
        chunks.extend(np.split(sub_chunk, sub_chunks_size / chunks_size))
    return np.array(chunks)


def append_image_chunks(np_arr, original_size):
    """appends a chucks from image to fit the to input layer of the nn.

    :param original_size: int, a size of original image. (rectangle [x,x])
    :param np_arr: str, an numpy array image chunks.
    :return: a list ot image chunks.
    :type: numpy array.
    """
    flat_img = np.append([], np_arr)
    return np.array(np.split(flat_img, original_size))


def save_image_from_np(image_np, to_path):
    """save image from numpy array.

    :param to_path: str, a path to save an image.
    :param image_np: numpy array, an image as numpy array..
    :return:
    """
    from matplotlib.image import imsave
    imsave(to_path, image_np.astype('uint8'))


def show_image_from_np(image_np):
    """show image from numpy array.

    :param image_np: numpy array, an image as numpy array..
    :return:
    """
    from matplotlib import pyplot as plt
    plt.figure()
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()


def draw_graph(folder_path, loss_error):
    """draw a graph of a loss function to screen and to file.

    :param folder_path: str, a path to folder for draw a graph.
    :return:
    """
    import matplotlib.pyplot as plt

    # Data for plotting
    points = np.asarray(loss_error)
    epoch = np.arange(0, len(loss_error))

    fig, ax = plt.subplots()
    ax.plot(epoch, points)

    ax.set(xlabel='epoch (num)', ylabel='loss function (sum)',
           title='Loss function over epoch.')
    ax.grid()

    fig.savefig("{}/{}.png".format(folder_path, datetime.now().strftime("%Y-%m-%d_%H-%M")))
    plt.show()


if __name__ == '__main__':
    from os import path

    # root folder path.
    root_folder = path.dirname(path.dirname(__file__))

    # # chunks size
    chunk_size = 16

    # split the image to small chunks
    data = split_image_chunks(image_path="{}/data/in/Photo_of_Lena_in_ppm.jpg".format(root_folder),
                              chunks_size=chunk_size)

    # fit data to input of the model.
    flat_train = fit_transform(data_to_fit=data) / 255.0

    seed(1)

    n_inputs = n_outputs = chunk_size * chunk_size
    network = initialize_network(n_inputs, round(n_inputs / 2), n_outputs)
    error = train_network(network=network, train=flat_train.tolist(), l_rate=0.3, n_epoch=100, n_outputs=n_outputs)

    draw_graph(folder_path="{}/data/out".format(root_folder), loss_error=error)

    prediction = [predict(network, row) for row in flat_train.tolist()]
    prediction = np.array(prediction) * 255.0

    # reverse fit transform
    pred_out_mult = reverse_transform(data_to_fit=prediction, size=chunk_size)

    # appends chunks to one image.
    img_data = append_image_chunks(np_arr=pred_out_mult, original_size=512).astype('uint8')

    # show image.
    show_image_from_np(image_np=img_data)

    # save image.
    save_image_from_np(image_np=img_data,
                       to_path="{0}/data/out/predicted_{1}.png"
                       .format(root_folder, datetime.now().strftime("%Y-%m-%d_%H-%M")))
