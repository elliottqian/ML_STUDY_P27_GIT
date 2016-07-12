# -*- coding: utf-8 -*-

import numpy as np

from machine_learning_algorithm import logistic_regression as lr


def zero_one_loss(label, output):
    """
    #0,1损失差不多的东西,正确返回1, 错误返回0
    """
    if label == output:
        return 1
    else:
        return 0


def h_theta(theta, new_x):
    """
    # theta为kx(n+1), k是种类数量, n是x的维度, x是mx(n+1)向量
    # z 是 k乘以m的向量, 没一列是每一个例子的和
    # 要按照列求和
    """
    z = np.dot(theta, new_x.T)
    z_list = lr.sigmoid(z)
    #print z_list
    z_sum = np.sum(z_list, axis=0)
    #print(z_sum)
    for i in range(z_list.shape[0]):
        z_list[:, i] = z_list[:, i] / z_sum[i]
    #print(z_list)
    return z_list


def p(i, j, mat_p):
    return mat_p[j][i]


def sgd_one_step(theta, new_x, label, alpha):
    """
    #根据公式来进行梯度下降的求解
    #这里y的值必须是从0开始的整数,例如 0,1,2,3
    """
    n = new_x.shape[1] - 1
    m = new_x.shape[0]
    k = theta.shape[0]
    for j in range(k):
        temp_sum = np.zeros((1, n + 1))
        mat_p = h_theta(theta, new_x)
        for index_x in range(m):
            temp_sum += new_x[index_x, :] * (zero_one_loss(label[index_x], j) - p(index_x, j, mat_p))
            print("----" + str(temp_sum))
        #print(theta[j, :].shape)
        #print((alpha * temp_sum).shape)
        theta[j, :] = theta[j, :] + alpha * temp_sum
        pass
    print("--------------theta---------------")
    print(theta)
    return theta


def sgd(theta, new_x, label, alpha, step_num=1000):
    """
    #批量梯度下降
    """
    for step in range(step_num):
        theta = sgd_one_step(theta, new_x, label, alpha)




def get_loss():
    """
    #返回损失函数的大小
    """
    pass


def predict(test_new_x, theta):
    result = h_theta(theta, test_new_x) #返回k乘以m向量  我们要得到  1乘以m向量
    out_put = np.zeros( theta.shape[0])
    for i in range(result.shape[1]):
        temp = result[:, i]
        max_index = 0
        max_num = -1.0
        for j in range(len(temp)):
            if temp[j] > max_num:
                max_num = temp[j]
                max_index = j
        out_put[i] = max_index
    print(out_put)
    return out_put
    pass


class SoftMax(object):

    def __init__(self, X, y, out_n):
        self.X = X
        self.y = y
        self.m = len(X)
        self.in_n = X.shape[1]
        self.out_n = out_n
        self.W = np.random.random((self.in_n, out_n))
        self.b = np.random.random((1, out_n))

    def get_loss(self):
        z = self.X.dot(self.W) + self.b
        prob = np.exp(-z)
        print(prob)
        sum_ax_1 = np.sum(prob, axis=1)
        print(sum_ax_1)
        for index in range(self.m):
            prob[index] = prob[index] / sum_ax_1[index]
        print(prob)
        loss = np.zeros((self.m, 1))
        for index in range(self.m):
            label = self.y[index]
            loss[index] = prob[index][label]
        print(loss)
        loss = - np.sum(np.log(loss))
        print(loss)
        pass

    def sgd(self):
        grand = 0


    def get_output(self):
        z = self.X.dot(self.W) + self.b
        prob = np.exp(-z)
        print(prob)
        predict = np.argmax(prob, axis=1)
        print(predict)
        return predict

    def zero_one_loss(self, label, output):
        """
        # 0,1损失差不多的东西,正确返回1, 错误返回0
        """
        if label == output:
            return 1
        else:
            return 0

"""m=3 n=5"""
temp_x = np.array([
    [0, 1, 2, 1, 2],
    [2, 1, 8, 1, 2],
    [-6, -6, -7, -8, -9]])

temp_theta = np.array([[-1.0, 2, -2, 1, 2, 1],
                       [1, 2, 3, 4, 5, 6],
                       [1, 2, -2, 5, 2, 11]])

"""m=3"""
temp_y = np.array([0, 0, 3])

if __name__ == "__main__":

    soft_max = SoftMax(temp_x, temp_y, 4)
    print(soft_max.W)
    soft_max.get_output()
    soft_max.get_loss()

    # new_x = lr.get_new_x(temp_x)
    # print(new_x)
    #
    # h = h_theta(temp_theta, new_x)
    # print(h)
    #
    # sgd_one_step(temp_theta, new_x, temp_y, 1)
    # sgd_one_step(temp_theta, new_x, temp_y, 1)
    # sgd_one_step(temp_theta, new_x, temp_y, 1)
    # sgd_one_step(temp_theta, new_x, temp_y, 1)
    # sgd_one_step(temp_theta, new_x, temp_y, 1)
    # t = sgd_one_step(temp_theta, new_x, temp_y, 0.1)
    #
    # predict(new_x, t)

    pass
