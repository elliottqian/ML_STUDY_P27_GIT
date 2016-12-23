# -*- coding:utf-8 -*-

import jieba
import codecs


import numpy as np
import matplotlib.pyplot as plt


class WordList(object):

    def __init__(self):

        self.word_list = []
        self.word_dict = {}

    def cut(self, path_name):
        """
        :return: 单词list, 单词个数的字典
        """
        file = codecs.open(path_name, 'rb', 'utf-8')
        for line in file.readlines():
            sentence = line.strip()
            # 先去掉首位空格, 然后吧迭代器转换成list
            word_itr = list(jieba.cut(sentence))
            self.word_list.extend(word_itr)
            for word in word_itr:
                if word in self.word_dict.keys():
                    self.word_dict[word] += 1
                else:
                    self.word_dict[word] = 1
        file.close()
        return self.word_list, self.word_dict

    def statistics(self):
        sorted_list = sorted(self.word_dict.items(), key=lambda item: item[1], reverse=True)
        for sl in sorted_list:
            print(sl)
        return sorted_list


class P(object):

    def __init__(self, x_list, y_list):

        self.x_l = x_list
        self.y_l = y_list

    def plot_histogram(self):

        n_groups = len(self.x_l)
        index = np.arange(n_groups)

        bar_width = 0.9
        rects1 = plt.bar(index, self.y_l, bar_width, color='b', label='Word')

        plt.xlabel('Group')
        plt.ylabel('Scores')
        plt.title('Scores by group and gender')

        plt.xticks(index + bar_width, self.x_l)

        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_line_chart(self):
        plt.figure(figsize=(16, 10))
        plt.plot(range(len(self.y_l)), self.y_l, label="$sin(x)$", color="red", linewidth=2)
        plt.legend()

        """展示"""
        plt.show()

"""



n_groups = 5

means_men = (20, 19, 18, 17, 16)
means_women = (25, 32, 34, 20, 25)

fig, ax = plt.subplots()
index = np.arange(n_groups)
# 宽度


opacity = 0.4
# 参数分别是:
# 坐标, 坐标对应的数组, 宽度, 透明度, 颜色, 标题

rects2 = plt.bar(index + bar_width, means_women, bar_width, alpha=opacity, color='r', label='Women')

plt.xlabel('Group')
plt.ylabel('Scores')
plt.title('Scores by group and gender')

# x轴标题位置
plt.xticks(index + bar_width, ('A', 'B', 'C', 'D', 'E'))

plt.ylim(0, 40)
plt.legend()

plt.tight_layout()
plt.show()

"""

file_path = "/home/elliottqian/鹿鼎记.txt"

if __name__ == "__main__":
    wl = WordList(file_path)
    wlist, wdict = wl.cut()
    for x in wlist:
        print(x)
    print(len(wlist))

    for x in wdict:
        print(x, wdict[x])

    sorted_list = wl.statistics()

    x = list(map(lambda a: a[0], sorted_list))
    y = list(map(lambda a: a[1], sorted_list))

    # x = ['A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E']
    # y = [25, 32, 34, 20, 25, 25, 32, 34, 20, 25, 25, 32, 34, 20, 25]
    pp = P(x, y)
    pp.plot_line_chart()

    # plt.figure(figsize=(16, 10))
    # plt.plot(range(len(y)), y, label="$sin(x)$", color="red", linewidth=2)
    # plt.legend()
    #
    # """展示"""
    # plt.show()
    pass
