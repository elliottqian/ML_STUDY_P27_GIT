# -*- coding: utf-8 -#-

import jieba

"""
扫描一遍,如果在dict中,dict的值就是位置,如果不在dict中,加入dict,位置就是dict最后一个(长度)
"""


def sentence2vector(sentence):
    word_position = {}
    word_list = jieba.cut(sentence)
    for word in word_list:
        if word not in word_position.keys():
            word_position[word] = len(word_position)
    return word_position
    pass


if __name__ == "__main__":
    test_sentence = "江南近海滨的一条大路上，一队清兵手执刀枪，押着七辆囚车，冲风冒寒，向北而行。"
    print(sentence2vector(test_sentence))
    pass