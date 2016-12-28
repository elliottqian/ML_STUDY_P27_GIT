# -*- coding: utf-8 -#-

from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.classification import LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint

import os
import sys

from data.x import OneDigit
from data.x import test_dir
from data.x import train_dir

sys.path.append("/home/elliottqian/Codes/Python/ML_STUDY_P27_GIT/data")
sys.path.append("/home/elliottqian/Codes/Python/ML_STUDY_P27_GIT/data/x.py")
print(sys.path)

os.environ["SPARK_HOME"] = "/home/elliottqian/d/Programs/spark_2.0.2_2.6"
PYSPARK_PYTHON = "/home/elliottqian/d/ubuntu/anaconda3/bin/python"
os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON

sys.path.append("/home/elliottqian/Codes/Python/ML_STUDY_P27_GIT/data")


class LabelPointData(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.labeled_point_data = []
        pass

    def covert(self):
        index = 0
        for label in self.y:
            labeled_point = LabeledPoint(label, self.X[index])
            self.labeled_point_data.append(labeled_point)
            index += 1
        pass


def check(test_data: OneDigit, logistic_regression_model: LogisticRegressionModel):
    num = len(test_data.X01)
    right_num = 0
    i = 0
    for x in test_data.X01:
        p_y = logistic_regression_model.predict(x)
        y = int(test_data.y01[i])
        i += 1
        if p_y == y:
            right_num += 1
    print(right_num)
    print(num)
    print(right_num / num)


if __name__ == "__main__":

    # 预处理部分
    train_digits = OneDigit(train_dir)
    train_digits.get_all_file_paths()
    train_digits.get_mat()
    train_digits.filter()

    lpd = LabelPointData(X=train_digits.X01, y=train_digits.y01)
    lpd.covert()
    print(lpd.labeled_point_data)

    conf = SparkConf().setAppName("logistic_regression_study").setMaster("local[4]")
    sc = SparkContext(conf=conf)

    lrm = LogisticRegressionWithLBFGS.train(sc.parallelize(lpd.labeled_point_data), iterations=10)

    # lrm.save(sc, "/home/elliottqian/lrm.model")
    # 预测部分

    # 1.检验数据读取和格式转换
    test_digits = OneDigit(test_dir)
    test_digits.get_all_file_paths()
    test_digits.get_mat()
    test_digits.filter()

    # 2.验证
    # lrm.predict([1.0, 0.0])
    check(test_digits, lrm)
    pass
