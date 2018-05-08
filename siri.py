import numpy as np
import random
import matplotlib.pyplot as plt

X = np.array([[1, 2], [3, 2], [3, 4], [7, 2], [10, 1], [7, 3], [11, 4], [13, 3]])
y = np.array([1, 1, 1, -1, -1, -1, -1, -1])
W = np.random.normal(size=2)
b = 0


def loss(x_sample, y_sample, weight, bias, c):
    svm_loss = (0.5 * np.dot(weight, weight)) + \
    c * np.sum(np.maximum(0, 1 - (y_sample * (np.matmul(x_sample, weight) + bias))), axis=0)
    return svm_loss


def gradient(x_sample, y_sample, weight, bias):
    v = 1 - y_sample * (np.dot(weight, x_sample) + bias)
    if v > 0:
        grad = weight -(y_sample * x_sample)
        bias = -y_sample
    else:
        grad = weight
        bias = 0
    return grad, bias


def sgd(x_input, y_label, weight, bias, c, alpha, n_iterations):
    while n_iterations > 0:
        index = random.sample(range(0, 8), 8)
        loss_updated = loss(x_input, y_label, weight, bias, c)
        print("Number of Iterations: ", n_iterations)
        print("Loss: ", loss_updated)
        print("-----------------------------")
        for i in index:
            (x_sample, y_sample) = list(zip(x_input, y_label))[i]
            grad_w, grad_b = gradient(x_sample, y_sample, weight, bias)
            weight -= alpha * grad_w
            bias -= alpha * grad_b
        n_iterations -= 1
    return weight, bias


weight, bias = sgd(X, y, W, b, 1, 0.001, 10000)
norm_weight = np.linalg.norm(weight)
margin = 1/norm_weight

def plot(x_input, y_label, weights, bias):
    y =