
import numpy as np
import random
import matplotlib.pyplot as plt

X = np.array([[1, 2], [3, 2], [3, 4], [7, 2], [10, 1], [7, 3], [11, 4], [13, 3]])
y = np.array([1, 1, 1, -1, -1, -1, -1, -1])
W = np.random.normal(size=2)
b = 0


def loss(x_sample, y_sample, weight, bias, c):
    '''
    Loss function
    :param x_sample: input x vector
    :param y_sample: input y vecotr
    :param weight: weight vector
    :param bias: bias scalar
    :param c: hyperparameter scalar
    :return: loss value scalar
    '''
    svm_loss = (0.5 * np.dot(weight, weight)) + \
               c * np.sum(np.maximum(0, 1 - (y_sample * (np.matmul(x_sample, weight) + bias))), axis=0)
    return svm_loss


def gradient(x_sample, y_sample, weight, bias):
    '''
    Calculates the gradient
    :param x_sample: input x vector
    :param y_sample: input y vector
    :param weight: weight vector
    :param bias: bias scalar
    :return: gradient scalar
    '''
    v = 1 - y_sample * (np.dot(weight, x_sample) + bias)
    if v > 0:
        grad = weight - (y_sample * x_sample)
        bias = -y_sample
    else:
        grad = weight
        bias = 0
    return grad, bias


def sgd(x_input, y_label, weight, bias, c, alpha, n_iterations):
    '''
    Performs stochastic gradient descent for n_iterations
    :param x_input: input x vector
    :param y_label: input y vector
    :param weight: weight vector
    :param bias: bias scalar
    :param c: hyper parameter c
    :param alpha: learning rate scalar
    :param n_iterations: number of iterations scalar
    :return: trained weights and biases
    '''
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


def supportvectors(x_input, y_label, weight_, bias_):
    return (1 - y_label * (np.dot(weight_, x_input) + bias_))


weight, bias = sgd(X, y, W, b, 1, 0.0001, 100000)
print("weight =", weight)
print("bias =", bias)

w1, w2 = weight[0], weight[1]

norm_weight = np.linalg.norm(weight)
margin = 1 / norm_weight

yy = np.linspace(0, 15)
xx = - bias / weight[0] - yy * weight[1] / weight[0]
margin_left = xx + 1 / weight[0]
margin_right = xx - 1 / weight[0]
axes = plt.gca()
axes.set_xlim([0, 15])
axes.set_ylim([0, 15])

plt.plot(xx, yy)
plt.plot(margin_left, yy, '--', color='orange')
plt.plot(margin_right, yy, '--', color='orange')
support_vectors = []
slack_variables = {}
for (x, y_label) in zip(X, y):
    if y_label == 1:
        plt.scatter(x[0], x[1], marker='+', color='blue')
    elif y_label == -1:
        plt.scatter(x[0], x[1], marker='o', color='red')
    slack = supportvectors(x, y_label, weight, bias)
    print(slack)
    if int(slack) == 0:
        support_vectors.append(x)
    slack_variables[str(x)] = max(0, slack)
print('support vectors =', support_vectors)
print('slackvariables =', slack_variables)
plt.show()
