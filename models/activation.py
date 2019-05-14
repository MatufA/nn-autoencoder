#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1. - x)


def relu(x):
    return x * (x > 0)


def relu_derivative(x):
    return 1. * (x > 0)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0 - x ** 2


