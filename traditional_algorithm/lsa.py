# -*- coding:utf-8 -*-

import numpy as np
"""
http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
"""


def svd(matrix):
    U, s, V = np.linalg.svd(matrix, full_matrices=False)
    S = np.diag(s)


if __name__ == "__main__":
    test_matrix = np.array([[5, 3, 2.5, 0, 0, 0, 0],
                            [2, 2.5, 5, 2, 0, 0, 0],
                            [2, 0, 0, 4, 4.5, 0, 5],
                            [5, 0, 3, 4.5, 0, 4, 0],
                            [4, 3, 2, 4, 3.5, 4, 0]])

    U, s, V = np.linalg.svd(test_matrix, full_matrices=False)

    print(U.shape)
    print(V.shape)
    print U
    print(s)
    print(V)

    S = np.diag(s)
    #S[2][2] = 0
    #S[3][3] = 0
    S[4][4] = 0
    print(S)
    print np.dot(U, np.dot(S, V))

    pass


"""
[[ 44.   31.5  39.   33.5  15.5  18.    5. ]
 [ 45.5  32.5  41.5  36.   15.5  20.5   4. ]
 [ 40.   18.5  24.5  38.   26.   16.5  15.5]
 [ 63.   37.   53.5  55.   26.   33.    9.5]
 [ 68.   42.5  56.5  59.   32.   34.5  11.5]]
"""