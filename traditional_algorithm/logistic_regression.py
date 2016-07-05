# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def h_theta(new_x, theta):
    z = (new_x * theta).sum(axis=1)
    return sigmoid(z)


def error(x, y, theta):
    print(np.log(h_theta(x, theta)))
    temp_ones = np.ones((1, x.shape[0]))
    print(temp_ones)
    error = y * np.log(h_theta(x, theta)) + (temp_ones - y) * np.log(temp_ones - h_theta(x, theta))
    print(h_theta(x, theta))
    print(temp_ones - h_theta(x, theta))
    return error
    pass


def get_new_x(x):
    ones_array = np.ones((x.shape[0], 1))
    new_x = np.hstack((ones_array, x))
    return new_x


def batch_gradient_descent(new_x, y, alpha, theta, step_num=5000):
    """
    #批量梯度下降
    :param new_x:加入x0的新的x矩阵
    :param y:y向量
    :param alpha:学习速度
    :param theta:最后需要的权重参数
    :return:权重参数
    """
    m = np.shape(new_x)[0]
    i = 0
    while True:
        temp_1 = h_theta(new_x, theta) - y
        #print(temp_1)
        temp_2 = np.dot(temp_1, new_x)
        #print("temp_2:")
        #print(temp_2)
        # print(alpha * temp_2)
        theta = theta - alpha * temp_2
        #print(np.sum(np.abs((alpha / m) * temp_2)))
        i += 1
        if i >= step_num:
            break
    print(theta)
    return theta


def read_a_text(text_path):
    x_vector = []
    data = open(text_path)
    for line in data.readlines():
        x_vector.extend(line.strip())
    x_vector = map(lambda t: int(t), x_vector)
    return np.array(x_vector)


def read_text_label(text_path):
    y = int(text_path.split("_")[0].split("\\")[-1])
    return y


def create_train_x(dir_path):
    files = os.listdir(dir_path)
    train_x = []
    label = []
    for f in files:
        full_path = dir_path + os.sep + f
        temp = read_a_text(full_path)
        train_x.append(temp)
        y = read_text_label(full_path)
        label.append(y)
    train_x_vector = np.array(train_x)
    train_y_vector = np.array(label)
    return train_x_vector, train_y_vector


def predict(theta, x):
    z = (x * theta).sum(axis=1)
    sig = sigmoid(z)
    result_list = []
    for s in sig:
        if s >= 0.5:
            result_list.append(1)
        else:
            result_list.append(0)
    return result_list


def get_precision(predict_list, test_list):
    length = len(predict_list)
    correct = 0
    for index in range(length):
        if predict_list[index] == test_list[index]:
            correct += 1
        else:
            print("error index:" + str(index))
    return correct / float(length)


class LogisticRegression(object):

    def __init__(self):
        pass


"""m=3 n=5"""
temp_x = np.array([
    [0, 1, 2, 1, 2],
    [2, 1, 8, 1, 2],
    [-6, -6, -7, -8, -9]])

temp_theta = np.array([-1, 2, -2, 1, 2, 1])

"""m=3"""
temp_y = np.array([1, 1, 0])


if __name__ == "__main__":

    # test_num = np.array([1, 2, 2])
    # test_num_2 = np.array([2, 3, 4])
    # htheta = np.array([1, 2, 3, 4])
    # print np.dot(test_num, test_num_2)
    # print(np.log(test_num))
    #
    # #print h_theta(test_num, htheta)
    #
    # test_num_sum = np.array([test_num, test_num_2])
    # print(test_num_sum)
    #
    # print test_num_sum * test_num
    # print (test_num_sum * test_num).sum(axis=1).T
    #
    #
    # print h_theta(temp_x, temp_theta)
    #
    # error(temp_x, temp_y, temp_theta)
    temp_new_x = get_new_x(temp_x)
    print(temp_new_x)
    print(temp_theta)
    #
    #
    #
    # print(temp_new_x.T)
    z = (temp_new_x * temp_theta).sum(axis=1)
    print("计算点乘的和")
    print(z)
    # test_batch_gradient_descent(temp_new_x, temp_y, 10, temp_theta)

    print("--------------------")

    theta = batch_gradient_descent(temp_new_x, temp_y, 0.6, temp_theta)

    y = theta * temp_new_x[1]
    print(h_theta(temp_new_x, theta))
    print(predict(temp_new_x, theta))

    print(get_precision(predict(temp_new_x, theta), temp_y))

    pass

