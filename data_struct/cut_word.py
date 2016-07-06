# -*- coding:utf-8 -*-

"""
使用字典树进行正向最大匹配分词
"""
import dict_tree


class CutWord(object):

    def __init__(self, dict_tree):
        self.dict_tree = dict_tree
        sentence_list = []
        pass

    def read_sentence(self):
        pass

    def cut_a_word_out(self, word_list, temp_sentence):
        """
        #从一句话里面分出第一个词
        :param word_list: 分出来的词列表
        :param temp_sentence: 剩余未分词的句子
        :return:分出来的词列表, 剩余未分词的句子
        """
        """从根节点开始"""
        temp_node = self.dict_tree.root

        """遍历一句话,去匹配最字典树"""
        for word_index in range(len(temp_sentence)):
            """看这个字在字典树中能不能继续匹配下去"""
            son_index = self.dict_tree.is_son_exist_char(temp_sentence[word_index], temp_node)
            """能继续匹配就继续匹配下去"""
            if son_index >= 0:
                temp_node = temp_node.son_list[son_index]
                """如果已经达到句子长度,就不匹配了,输出整个句子(只有一个词)"""
                if word_index == len(temp_sentence) - 1:
                    word_list.append(temp_sentence)
                    return word_list, []
            else:
                """如果第一个字就不在的化,就匹配一个字"""
                if word_index == 0:
                    word_list.append(temp_sentence[0])
                    new_sentence = temp_sentence[1:]
                    return word_list, new_sentence
                else:
                    """正常匹配步骤"""
                    word_list.append(temp_sentence[0:word_index])
                    new_sentence = temp_sentence[word_index:]
                    return word_list, new_sentence

    def cut_sentence(self, word_list, sentence):
        temp_word_list, temp_sentence = word_list, sentence
        while len(temp_sentence) > 0:
            temp_word_list, temp_sentence = self.cut_a_word_out(temp_word_list, temp_sentence)
        return temp_word_list

    def remove_punctuation(self):
        pass


if __name__ == "__main__":

    file_path = "../dictionary"

    sentence = u"我们一起去爬山"
    sentence2 = u"我们一起去爬山"
    test = u"我们"
    word_list = []

    my_dict_tree = dict_tree.DictionaryTree()
    my_dict_tree.read_file(file_path)
    my_dict_tree.create_dict_tree()
    cut_word = CutWord(my_dict_tree)

    # word_list, sentence = cut_word.cut_a_word_out(word_list, sentence)
    # print(word_list[0])
    # print(sentence)
    # word_list, sentence = cut_word.cut_a_word_out(word_list, sentence)
    # print(word_list[1])
    # print(sentence)
    # word_list, sentence = cut_word.cut_a_word_out(word_list, sentence)
    # print(word_list[2])
    # print(sentence)
    # word_list, sentence = cut_word.cut_a_word_out(word_list, sentence)
    # print(word_list[3])
    # print(sentence)
    #
    # word_list, sentence = cut_word.cut_a_word_out(word_list, u"哈")
    # print(word_list[-1])
    word_list = cut_word.cut_sentence(word_list, sentence)
    print("-------------------")
    for x in word_list:
        print(x)




    # wl = []
    # wl = cut_word.cut_sentence(wl, sentence2)
    #
    # for x in wl:
    #     print(x)

    pass
