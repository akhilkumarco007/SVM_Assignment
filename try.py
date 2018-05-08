import numpy as np
import random

X = np.array([[1, 2], [3, 2], [3, 4], [7, 2], [10, 1], [7, 3], [11, 4], [13, 3]])

y = np.array([1, 1, 1, -1, -1, -1, -1, -1])

W = np.random.uniform(size=2)

b = 0

loss = np.dot(W, X[0])

print loss

# m = np.shape(y)[0]
# index = random.randint(0, m-1)
#
# (x_, y_, w_, b_) = zip(X, y, W, b)[index]
# print x_, y_, w_, b_
# # x = X[index]
# # x_, y_, w_, b_ = zip(*random.sample(list(zip(X, y, W, b)), 1))
# # x_, y_, w_, b_ = x_[0], y_[0], w_[0], b_[0]
# loss = 0
# grad = 0
# #
# loss += sum(((w_ * w_)/2) + 0.5 * max(0, sum(1 - y_*(w_* x_ + b_))))
#
# print loss
#
# # c =  np.transpose(y) *((W * X) + b)
#
# # print c
# for (x_, y_, w_, b_) in zip(X, y, W, b):
#     v = y_*np.dot(w_, x_)
#     loss += sum(((w_ * w_)/2) + max(0, sum(1 - y_*(np.dot(w_, x_) + b_))))
#     if v >= 1:
#         grad += 0
#     else:
#         grad += sum(-y_*x_)


# loss = sum((np.multiply(W, W))/2) + sum(max(0, 1 - np.dot(y*((np.multiply(W, X) + b)))))

# loss = (np.dot(np.transpose(W), W))/2
# print loss