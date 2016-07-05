# -*- coding:utf-8 -*-

import logistic_regression as lr
import numpy as np


def change_label(x):
    if x == 9:
        return 0
    else:
        return 1



if __name__ == "__main__":

    data_dir = "D:\\MyWork\\data\\digits\\trainingDigits"
    train_x, train_y = lr.create_train_x(data_dir)
    print(train_y.shape)
    print(train_x.shape)

    theta = np.random.random(size=train_x.shape[1]+1) * 10 - 5
    print(theta)

    print(train_y[-1])
    train_y = map(lambda x: change_label(x), train_y)
    print(train_y[-1])

    temp_new_x = lr.get_new_x(train_x)
    w = lr.batch_gradient_descent(temp_new_x, train_y, 0.1, theta, 10)

    """2. 进行预测"""
    test_dir = "D:\\MyWork\\data\digits\\testDigits"
    #读取预测文件
    test_x, test_y = lr.create_train_x(test_dir)
    new_test_x = lr.get_new_x(test_x)

    test_y = map(lambda x: change_label(x), test_y)

    first = 100
    last = 150
    debug_x = new_test_x[first:last]
    debug_y = test_y[first:last]

    predict = lr.predict(w, debug_x)
    print(predict)
    print lr.get_precision(predict, debug_y)

    print lr.h_theta(debug_x, theta)

    predict = lr.predict(w, new_test_x)
    print(predict)
    print lr.get_precision(predict, test_y)

    pass
