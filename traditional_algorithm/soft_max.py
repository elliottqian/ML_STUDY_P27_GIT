# -*- coding: utf-8 -*-

import numpy as np

from machine_learning_algorithm import logistic_regression as lr


# def zero_one_loss(label, output):
#     """
#     #0,1损失差不多的东西,正确返回1, 错误返回0
#     """
#     if label == output:
#         return 1
#     else:
#         return 0
#
#
# def h_theta(theta, new_x):
#     """
#     # theta为kx(n+1), k是种类数量, n是x的维度, x是mx(n+1)向量
#     # z 是 k乘以m的向量, 没一列是每一个例子的和
#     # 要按照列求和
#     """
#     z = np.dot(theta, new_x.T)
#     z_list = lr.sigmoid(z)
#     #print z_list
#     z_sum = np.sum(z_list, axis=0)
#     #print(z_sum)
#     for i in range(z_list.shape[0]):
#         z_list[:, i] = z_list[:, i] / z_sum[i]
#     #print(z_list)
#     return z_list
#
#
# def p(i, j, mat_p):
#     return mat_p[j][i]
#
#
# def sgd_one_step(theta, new_x, label, alpha):
#     """
#     #根据公式来进行梯度下降的求解
#     #这里y的值必须是从0开始的整数,例如 0,1,2,3
#     """
#     n = new_x.shape[1] - 1
#     m = new_x.shape[0]
#     k = theta.shape[0]
#     for j in range(k):
#         temp_sum = np.zeros((1, n + 1))
#         mat_p = h_theta(theta, new_x)
#         for index_x in range(m):
#             temp_sum += new_x[index_x, :] * (zero_one_loss(label[index_x], j) - p(index_x, j, mat_p))
#             print("----" + str(temp_sum))
#         #print(theta[j, :].shape)
#         #print((alpha * temp_sum).shape)
#         theta[j, :] = theta[j, :] + alpha * temp_sum
#         pass
#     print("--------------theta---------------")
#     print(theta)
#     return theta
#
#
# def sgd(theta, new_x, label, alpha, step_num=1000):
#     """
#     #批量梯度下降
#     """
#     for step in range(step_num):
#         theta = sgd_one_step(theta, new_x, label, alpha)
#
#
# def get_loss():
#     """
#     #返回损失函数的大小
#     """
#     pass
#
#
# def predict(test_new_x, theta):
#     result = h_theta(theta, test_new_x) #返回k乘以m向量  我们要得到  1乘以m向量
#     out_put = np.zeros( theta.shape[0])
#     for i in range(result.shape[1]):
#         temp = result[:, i]
#         max_index = 0
#         max_num = -1.0
#         for j in range(len(temp)):
#             if temp[j] > max_num:
#                 max_num = temp[j]
#                 max_index = j
#         out_put[i] = max_index
#     print(out_put)
#     return out_put
#     pass


class SoftMax(object):

    def __init__(self, X, y, out_n):
        self.X = X
        self.y = y
        self.m = len(X)
        self.step_length = 0.01
        self.in_n = X.shape[1]
        self.out_n = out_n
        np.random.seed(0)
        self.W = np.random.random((self.in_n, out_n))
        self.b = np.random.random((1, out_n))

    def get_loss(self):
        """
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
        """
        prob = self.get_output()
        prob_v = np.zeros((len(self.y,)))
        print(prob)
        for i in range(len(self.y)):
            prob_v[i] = prob[i][self.y[i]]
        loss = np.sum(- np.log(prob_v)) / self.m
        return loss

    def gradient(self):
        """
        # 每一步的梯度
        # 公式中, d_theta是每一行的导数, 是一个向量 总共k个向量, k是类别
        # 理解梯度公式很重要
        """
        d_theta = np.zeros((self.in_n, self.out_n)).T  # k * n  k是种类数量, n是训练数据维度
        p = self.get_output()  # p 是输出概率矩阵, 每行对应每个样本的每种类别的矩阵,列是种类数 m * k
        print(p)
        for j in range(self.out_n):
            j_v = np.ones((len(self.y),)) * j
            zo = self.zero_one_vector_loss(j_v)
            print(zo)

            my_want = zo - p[:, j]  # 属于第j类的概率
            print("---------------- gradient ------------------")
            print(self.X)
            print(np.array([my_want]).T)
            temp = self.X * np.array([my_want]).T  # 这一步很重要, 避免了循环
            d_theta[j] = np.sum(temp, axis=0)
        return - d_theta  # 这里取负号, 不然越算越远了

    def sgd(self, setp_num):
        for i in range(setp_num):
            self.W -= self.step_length * self.gradient().T
        print(self.W)
        return self.W

    def get_output(self):
        """
        # 返回的是soft_max的概率
        """
        z = self.X.dot(self.W) + self.b
        z_max = np.max(z)
        # print("----------z------------------")
        # print(z - z_max)
        prob = np.exp(z - z_max)
        return prob / np.array([np.sum(prob, axis=1)]).T


    def predict(self):
        p = self.get_output()
        predict = np.argmax(p, axis=1)
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

    def zero_one_vector_loss(self, label_v):
        l = len(self.y)
        out = np.zeros((l,))
        for i in range(l):
            if self.y[i] == label_v[i]:
                out[i] = 1
            else:
                out[i] = 0
        return out


"""m=3 n=5"""
temp_x = np.array([
    [0, 1, 2, 1, 2],
    [2, 1, 8, 1, 2],
    [-2, -1, -1, -2, -1]])

temp_theta = np.array([[-1.0, 2, -2, 1, 2, 1],
                       [1, 2, 3, 4, 5, 6],
                       [1, 2, -2, 5, 2, 11]])

"""m=3"""
temp_y = np.array([1, 1, 3])

if __name__ == "__main__":

    soft_max = SoftMax(temp_x, temp_y, 4)
    #print(soft_max.W)
    soft_max.gradient()
    #soft_max.get_loss()

    #print(np.exp(1.27749379e+08))

    #label = [0, 1, 2]
    #label = label * 2
    #print(label)
    #print(soft_max.zero_one_vector_loss(label))

    # #soft_max.grand()
    soft_max.sgd(423)
    soft_max.predict()
    #
    print(soft_max.get_loss())
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
