# -*- coding: utf-8 -*-

"""
三层前馈神经网络
每层神经元个数分别为
l_1, l_2, l3
"""

import numpy as np

from machine_learning_algorithm import logistic_regression as lr


class Layer(object):

    def __init__(self, unit_num, input_num, activation=lr.sigmoid):
        self.unit_num = unit_num
        self.activation = activation
        self.input_num = input_num

        self.W = np.random.random(size=(self.unit_num, input_num)) - 0.5
        self.b_pre = np.random.random(size=self.unit_num) - 0.5


    def forward_with_input(self, input_data):
        """
        #前一层的输入
        :param input_data:
        :return:
        """
        self.z = np.dot(self.W, input_data.T) + self.b_pre
        self.a = self.activation(self.z)
        return self.a

    # def forward(self):
    #     return self.forward_with_input(self.input_data)

    def last_layer_residual_error(self, true_out_y):
        """每一个单元都有一个残差, 这个是计算最后一层的残差"""
        self.r_error = - (true_out_y - self.a) * self.a * (1 - self.a)
        return self.r_error

    def residual_error(self, next_b_re, next_W):
        """计算非最后一层的残差"""
        self.r_error = np.dot(next_W.T, next_b_re.T) * self.a * (1 - self.a)
        return self.r_error

    def J_W(self, input_data):
        """J对W求偏导数"""
        length_r_error = len(self.r_error)
        length_input_data = len(input_data)
        self.j_w = np.dot(self.r_error.reshape((length_r_error, 1)), input_data.reshape(1, length_input_data))
        return self.j_w

    def J_B(self):
        self.j_b = self.r_error
        return self.j_b


class NeuralNetwork(object):

    def __init__(self, layer_num_list, input_data, output_data):
        """
        :param layer_list: 层数列表 ,例如 [3, 4, 4], 表示第一层3个, 第二层4个, 第三层3个
        :param input_data: 输入一个实例的数据, 例如 [1, 2, 4]表示一个样本
        """
        self.input_data = input_data
        self.output_data = output_data
        self.layer_num = len(layer_num_list)
        self.layer_list = []
        """layer_list[0]代表第二层"""
        for i in range(self.layer_num - 1):
            self.layer_list.append(Layer(layer_num_list[i+1], layer_num_list[i]))

        self.W_list = []
        self.b_list = []


    def get_W_b_list(self):
        for layer in self.layer_list:
            self.W_list.append(layer.W)
            self.b_list.append(layer.b_pre)


    def forward(self):
        input_list = []
        input_list.append(self.input_data)
        i = 0
        for layer in self.layer_list:
            this_layer_input = input_list[i]
            this_layer_output = layer.forward_with_input(this_layer_input)
            input_list.append(this_layer_output)
            i += 1
        return input_list[-1]

    def back_forward(self, y):
        """先计算输出层的残差,以此计算前面各层的残差,计算"""
        last_residual_error = self.layer_list[-1].last_layer_residual_error(y)
        print last_residual_error
        for i in range(1, len(self.layer_list)):
            layer_index = len(self.layer_list) - 1 - i
            print "layer_index:" + str(layer_index)
            layer = self.layer_list[layer_index]
            next_layer = self.layer_list[layer_index+1]
            layer.residual_error(next_layer.r_error, next_layer.W)

    def J_W_B(self):
        input = self.input_data
        for layer in self.layer_list:
            layer.J_W(input)
            layer.J_B()



    def batch_gradient_descent(self):
        """神经网络的批量梯度下降"""


        pass


class TrainNetwork(object):

    def __init__(self, network, train_x, train_y):
        self.network = network
        self.train_x = train_x
        self.train_y = train_y
        self.m = np.shape(train_x)[0]
        pass

    def batch_gradient_descent(self):
        for index in range(self.m):
            train_x_row, train_y_row = self.train_x[index], self.train_y[index]

            pass
        pass

    def get_a_input_and_train(self, input, output):
        self.network.input_data = input
        self.network.output_data = output
        self.network.forward()
        self.network.back_forward(output)
        self.network.J_W_B()
        self.network.get_W_b_list()
        pass


if __name__ == "__main__":
    input_data = np.array([1, 2, 3])
    input_data_list = np.array([[1, 2, 3], [2, 4, 3]])
    test_y = np.array([3, 1, 4, 2])
    w = np.array([[1, 2, 3],
                  [4, 5, 6]])
    print(np.dot(w, input_data.T))

    layer = Layer(4, 3)
    print layer.forward_with_input(input_data)
    print layer.W
    layer.last_layer_residual_error(test_y)
    print layer.r_error

    print layer.J_W(input_data)
    print layer.J_B()
    #print layer.residual_error(test_y)

    #神经网络的测试

    print "--------------------network test--------------------------"
    nn = NeuralNetwork([3, 4, 3], input_data)

    print nn.forward()

    print nn.layer_list[0].a
    print nn.layer_list[1].a

    nn.back_forward(np.array([3, 1, 4]))

    print "--------------------r_error--------------------------"
    print nn.layer_list[0].r_error
    print nn.layer_list[1].r_error

    print "--------------------偏导数--------------------------"
    a = np.array([1, 2, 3, 4])
    b = np.array([1, 2, 3])
    print(len(a))
    print np.dot(a.reshape((4, 1)), b.reshape(1, 3))
    nn.J_W_B()
    pass
