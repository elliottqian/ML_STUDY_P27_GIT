# -*- coding:utf-8 -*-

import nltk


def test_1():
    sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""
    sentence_2 = u"""我们要回家了"""
    tokens = nltk.word_tokenize(sentence_2)
    print(tokens)
    pass


if __name__ == u"__main__":
    test_1()
    pass
