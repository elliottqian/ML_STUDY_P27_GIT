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

    weight = []

    def __init__(self, length):
        self.length = length
        self.weight = [0] * length

    def word_list_to_vector(self, word_list: list, word_position: dict):
        """
        # 把词列表添加到原有向量里面
        :return:
        """
        for word in word_list:
            p = word_position[word]
            self.weight[p] += 1
        pass
    pass


if __name__ == u"__main__":
    # test_word = ['a', 'c', 'a', 'c', '吃饭']
    # wp = WordPosition()
    # wp.word_list_to_vector(test_word)

    file_path = "/home/elliottqian/鹿鼎记.txt"

    wl = WordList(file_path)
    wl.cut()
    word_dict = wl.statistics()

    wp = WordPosition(top=0.1)
    word_position, l = wp.word_list_to_vector(word_dict)

    print(l)
    pass