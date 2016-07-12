# -*- coding: utf-8 -*-

import sklearn
import numpy as np
import matplotlib
from sklearn import datasets
import matplotlib.pyplot as mpp
import sklearn.linear_model


u"""
# 编写一个自己的前馈神经网络,
# 输入是2维
# 输出是2维
# 中间隐含层可以自己选择
# 总共3层
"""


def test_1():
    # 读取sk里面的数据, 并且绘图
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    print(X)
    mpp.scatter(X[:,0], X[:,1], s=40, c=y)
    #mpp.plot(X[:,0], X[:,1])
    mpp.show()
    return X, y


def sk_logistic_regression(X, y):
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X, y)


# Helper function to evaluate the total loss on the data set
def calculate_loss(X, model, num_examples, reg_lambda):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    print(a1)
    z2 = a1.dot(W2) + b2
    print(z2)
    exp_scores = np.exp(z2)
    print(exp_scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    print(probs)
    # Calculating the loss
    print(probs[[0, 1, 2, 3], 0])
    corect_logprobs = -np.log(probs[range(num_examples), y])
    #print(corect_logprobs)
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss


# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(X, nn_hdim, nn_input_dim, nn_output_dim, num_passes=20000, print_loss=False):

    # 训练集大小
    num_examples = len(X)

    # Gradient descent parameters (I picked these by hand)
    # learning rate for gradient descent
    epsilon = 0.01
    # regularization strength
    reg_lambda = 0.01

    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    # W的初始化要按照一定的规则, tanh和sigmo有不同的初始规则
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in xrange(0, num_passes):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
          print "Loss after iteration %i: %f" %(i, calculate_loss(model))

    return model


def nn():
    # 训练集大小
    num_examples = len(X)
    # input layer dimensionality
    nn_input_dim = 2
    # output layer dimensionality
    nn_output_dim = 2

    # Gradient descent parameters (I picked these by hand)
    # learning rate for gradient descent
    epsilon = 0.01
    # regularization strength
    reg_lambda = 0.01



if __name__ == u"__main__":

    X, y = datasets.make_moons(200, noise=0.20)

    nn_input_dim = 2
    nn_hdim = 3
    nn_output_dim = 2
    num_examples = len(X)
    epsilon = 0.01
    reg_lambda = 0.01

    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    model = {}
    model['W1'] = W1
    model['b1'] = b1
    model['W2'] = W2
    model['b2'] = b2

    calculate_loss(X, model, num_examples, reg_lambda)

    #
    # nn_input_dim = 2
    # nn_hdim = 3
    # b1 = np.zeros((1, nn_hdim))
    # W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    #
    # print(W1)
    #
    # print(X.dot(W1) + b1)
    # print(X.dot(W1).shape)
    # print(b1.shape)
    pass
