# -*- coding: utf-8 -*-

import numpy as np


def similarity(item_id_1, item_id_2):
    """
    #这里计算的是同现相似度
    """
    sim = 0
    for key in item_id_1.keys():
        if item_id_2.has_key(key):
            sim += 1
    return sim


def get_similarity_matrix(items_user_list, similarity_fun=similarity):
    """
    #计算相似度矩阵,这是一个对称矩阵
    """
    item_num = len(items_user_list)
    similarity_matrix = np.zeros((item_num, item_num))
    for i in range(item_num):
        for j in range(i, item_num):
            sim = similarity_fun(items_user_list[i], items_user_list[j])
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim
    return similarity_matrix


def get_items_user_list(score_matrix):
    """
    #建立倒排索引表,
    :param score_matrix:用户,评分矩阵
    :return:物品评分用户倒排索引表
    """
    (user_num, item_num) = score_matrix.shape
    items_user_list = []
    for i in range(item_num):
        temp_set = {}
        for u in range(user_num):
            if score_matrix[u][i] >= 0.1:
                temp_set[u] = score_matrix[u][i]
        items_user_list.append(temp_set)
    return items_user_list


def get_all_user_recommend(score_matrix, similarity_matrix):
    """
    #得到最后的推荐矩阵
    """
    result = np.dot(score_matrix, similarity_matrix)
    return result


if __name__ == "__main__":
    test_matrix = np.array([[5, 3, 2.5, 0, 0, 0, 0],
                            [2, 2.5, 5, 2, 0, 0, 0],
                            [2, 0, 0, 4, 4.5, 0, 5],
                            [5, 0, 3, 4.5, 0, 4, 0],
                            [4, 3, 2, 4, 3.5, 4, 0]])
    (a, b) = test_matrix.shape
    print(a, b)

    it_us_l = get_items_user_list(test_matrix)
    print(it_us_l[0])
    print(it_us_l[1])

    print similarity(it_us_l[0], it_us_l[1])

    similarity_matrix = get_similarity_matrix(it_us_l)
    print similarity_matrix

    print(get_all_user_recommend(test_matrix, similarity_matrix))

    pass
