# -*- coding:utf-8 -*-


class OneGram(object):
    pass


def get_p(word, sentence_list):
    denominator = len(sentence_list)
    molecule = 0
    for sentence in sentence_list:
        if sentence.__contains__(word):
            molecule += 1
    return molecule / float(denominator)


def one_gram_cut():
    pass


def make_word_p_dict(word_dict, sentence_list):
    l = float(len(sentence_list))
    for sentence in sentence_list:
        for word in word_set:
            if sentence.__contains__(word):
                word_dict[word] += 1 / l
    return word_dict


def make_word_set(sentence_list):
    word_dict = {}
    for sentence in sentence_list:
        for i in range(len(sentence)):
            word_dict[sentence[i]] = 0
            if i + 1 < len(sentence):
                word_dict[sentence[i] + sentence[i+1]] = 0
            if i + 2 < len(sentence):
                word_dict[sentence[i] + sentence[i+1] + sentence[i+2]] = 0
            if i + 3 < len(sentence):
                word_dict[sentence[i] + sentence[i+1] + sentence[i+2] + sentence[i+3]] = 0
        pass
    return word_dict


def cut_word_test(word_dict, xx):
    pass

if __name__ == "__main__":
    sentence_list = [u"我要去吃饭", u"我们要吃饭", u"我喊他去吃饭", u"去食堂吃饭很痛苦"]
    print(get_p(u"我们", sentence_list))

    word_set = make_word_set(sentence_list)

    word_dict = make_word_p_dict(word_set, sentence_list)

    for x in word_dict:
        if word_dict[x] > 0.4:
            print(x)
            print(word_dict[x])
    pass
