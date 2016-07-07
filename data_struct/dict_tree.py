# -*- coding: utf-8 -*-


"""
an implementation of the dictionary tree
for chinese
"""

import codecs


class Node(object):

    def __init__(self, character):
        self.character = character
        self.son_list = []

    def son_num(self):
        return len(self.son_list)


class DictionaryTree(object):

    def __init__(self):
        self.dictionary = set()
        self.root = Node(None)
        pass

    def read_file(self, dict_path):
        """
        #从文本文件中读取字典,返回字典集合
        :param dict_path: 字典文件路径
        :return: 一个字典set
        """
        dict_file = codecs.open(dict_path, "r", "utf-8")
        for word in dict_file.readlines():
            self.dictionary.add(word.strip())
            pass
        return self.dictionary

    def update_dict_tree_with_dict(self, dictionary):
        self.dictionary = dictionary
        self.create_dict_tree()

    def create_dict_tree(self):
        for word in self.dictionary:

            # 循环从字典里面取词
            # 每次从根节点插入新词,所以有 temp_root = self.root-
            temp_root = self.root

            # 插入字典代码
            for i in range(len(word)):  # 对每一个单词来循环
                """先检查temp的子节点有没有这个字,没有就加入新的节点,"""
                exist = self.is_son_exist_char(word[i], temp_root)
                """移动temp节点,直到叶子节点"""
                if exist >= 0:
                    temp_root = temp_root.son_list[exist]
                else:
                    new_node = Node(word[i])
                    temp_root.son_list.append(new_node)
                    temp_root = temp_root.son_list[-1]

    def is_son_exist_char(self, character, node):
        """
        #检查当前字符在不在子节点,在的花,返回节点的index,不在返回-1
        """
        for i in range(len(node.son_list)):
            if node.son_list[i].character == character:
                return i
        return -1


# def test_is_son_exist_char(instance):
#     test_char = "AB"[0]
#     print(test_char)
#
#     test_node = Node(None)
#     son_node_1 = Node("AB"[0])
#     son_node_2 = Node("AB"[1])
#     test_node.son_list.append(son_node_1)
#     test_node.son_list.append(son_node_2)
#
#     print instance.is_son_exist_char(test_char, test_node)


if __name__ == "__main__":
    file_path = "../dictionary"
    dict_tree = DictionaryTree()

    dictionary = dict_tree.read_file(file_path)
    dict_tree.create_dict_tree()
    new_dict_ste = set()
    new_dict_ste.add(u"音乐")
    new_dict_ste.add(u"音调")
    dict_tree.update_dict_tree_with_dict(new_dict_ste)

    for x in dict_tree.root.son_list:
        print(x.character)
        for y in x.son_list:
            print(y.character)

    pass
