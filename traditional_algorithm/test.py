# -*- coding: utf-8 -*-

import pandas as pd

import numpy as np
import string, os, sys


text_path = "D:\\MyWork\\data\\digits\\trainingDigits\\1_0.txt"


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


def create_test_x(dir_path):
    files = os.listdir(dir_path)
    train_x = []
    for f in files:
        full_path = dir_path + os.sep + f
        temp = read_a_text(full_path)
        train_x.append(temp)
    train_x_vector = np.array(train_x)
    return train_x_vector


if __name__ == "__main__":
    x, y = create_train_x("D:\\MyWork\\data\\digits\\trainingDigits")
    # x = read_a_text(text_path)
    # print(type(x[0]))
    # p = "D:\\MyWork\\data\\digits\\trainingDigits\\1_0.txt"
    # print int(p.split("_")[0].split("\\")[-1])
    print(x)
    print(y)
    pass


"""


dir = '/var'
print '----------- no sub dir'

files = os.listdir(dir)
for f in files:
    print dir + os.sep + f

print '----------- all dir'

for root, dirs, files in os.walk(dir):
    for name in files:
        print os.path.join(root, name)

"""