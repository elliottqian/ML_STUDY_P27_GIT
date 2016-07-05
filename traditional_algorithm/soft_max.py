# -*- coding: utf-8 -*-

import numpy as np
import logistic_regression as lr


def zero_one_loss(label, output):
    if label == output:
        return 1
    else:
        return 0


def h_theta(theta, new_x):
    z = np.dot(theta, new_x.T)
    z_list = lr.sigmoid(z)
    print z_list
    z_sum = np.sum(z_list)
    print(z_sum)
    return z_list / z_sum


"""m=3 n=5"""
temp_x = np.array([
    [0, 1, 2, 1, 2],
    [2, 1, 8, 1, 2],
    [-6, -6, -7, -8, -9]])

temp_theta = np.array([[-1, 2, -2, 1, 2, 1],
                       [1, 2, 3, 4, 5, 6],
                       [1, 2, -2, 5, 2, 11]])

"""m=3"""
temp_y = np.array([1, 1, 0])


def get_loss():
    pass


if __name__ == "__main__":
    new_x = lr.get_new_x(temp_x)
    print(new_x)

    h = h_theta(temp_theta, new_x)
    print(h)

    pass
