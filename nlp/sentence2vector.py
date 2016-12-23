# -*- coding:utf-8 -*-


"""
先讲需要表达的语料扫描一遍，
取 top 30% 作为 represent

然后再扫描一次， 转换成represent

复杂的O(2n)
"""

import sys
from nlp.jieba_cut import WordList

sys.path.append("~/nlp")


class WordPosition(object):

    def __init__(self, top=0.3):
        self.top = top
        self.word_position = {}

    def word_list_to_vector(self, word_statistic):
        """
        # 扫描所有语料
        :param word_list:
        :return:
        """
        top_index = int(len(word_statistic) * self.top)
        for i in range(len(word_statistic)):
            if i < top_index:
                self.word_position[word_statistic[i][0]] = i
            else:
                self.word_position[word_statistic[i][0]] = top_index
        return self.word_position, top_index + 1


class WordVector(object):

    def __init__(self, length: int, word_position: dict):
        self.word_position = word_position
        self.length = length
        self.weight = []

    def word_list_to_vector(self, word_dict: list):
        """
        # 把词列表添加到原有向量里面
        :return:
        """
        self.weight = [0] * self.length
        for word in word_dict:
            p = self.word_position[word]
            self.weight[p] += word_dict[word]
        return self.weight


def my_word_2_vector(path_list, top=0.1):

    # 用WordList 对象得到 词数统计
    wl = WordList()
    for path in path_list:
        wl.cut(path)
    word_dict = wl.statistics()

    # 得到单词位置和长度
    wp = WordPosition(top=top)
    word_position, l = wp.word_list_to_vector(word_dict)
    wv = WordVector(l, word_position)

    # 重置后从新统计
    w_list = []
    for path in path_list:
        wl.reset()
        temp_wd = wl.cut(path)

        # 根据单词位置的字典, 获得向量
        w = wv.word_list_to_vector(temp_wd)
        w_list.append(w)

    return w_list


if __name__ == u"__main__":

    # 用WordList 对象得到 词数统计
    wl = WordList()
    wl.cut("/home/elliottqian/a")
    wl.cut("/home/elliottqian/b")
    word_dict = wl.statistics()

    # 得到单词位置和长度
    wp = WordPosition(top=0.8)
    word_position, l = wp.word_list_to_vector(word_dict)

    print(l)
    print(word_position)

    # 重置后从新统计
    wl.reset()
    print(wl.word_dict)
    wd_01 = wl.cut("/home/elliottqian/a")
    print(wd_01)
    wl.reset()
    wd_02 = wl.cut("/home/elliottqian/b")
    print(wd_02)

    # 根据单词位置的字典, 获得向量
    wv = WordVector(l, word_position)
    w1 = wv.word_list_to_vector(wd_01)
    w2 = wv.word_list_to_vector(wd_02)

    print(w1)
    print(w2)


    wl = my_word_2_vector(["/home/elliottqian/a", "/home/elliottqian/b"], top=0.8)
    for w in wl:
        print(w)
    pass