from collections import Counter
import numpy as np
from math import sqrt

def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Not vectorized implementation.
    """
    x = x.tolist()
    n = len(x)
    m = len(x[0])
    res = 1
    for i in range(0, min(n, m)):
        if x[i][i] != 0:
            res *= x[i][i]
    return res


def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Not vectorized implementation.
    """

    x = x.tolist()
    y = y.tolist()
    return Counter(x) == Counter(y)


def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Not vectorized implementation.
    """

    x = x.tolist()
    n = len(x)
    mx = -1
    for i in range(1, n):
        if (x[i - 1] == 0 and x[i] > mx):
            mx = x[i]
    return mx


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Not vectorized implementation.
    """
    
    img = img.tolist()
    coefs = coefs.tolist()
    h, w = len(img), len(img[0])
    result = [[0 for _ in range(w)] for __ in range(h)]
    for i in range(h):
        for j in range(w):
            result[i][j] = img[i][j][0] * coefs[0] + img[i][j][1] * coefs[1] + img[i][j][2] * coefs[2]
    return np.array(result).astype(np.uint8)


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Not vectorized implementation.
    """
    
    x = sorted(x.tolist()) + [-1]
    elements = []
    counters = []
    i = 0
    cnt = 0
    while i < len(x) - 1:
        if x[i] == x[i + 1]:
            cnt += 1
        else:
            elements.append(x[i])
            counters.append(cnt + 1)
            cnt = 0
        i += 1
    return (elements, counters)


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Not vectorized implementation.
    """

    x = x.tolist()
    y = y.tolist()
    n = len(x)
    z = [[0 for i in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            z[i][j] = sqrt((x[i][0] - y[j][0]) ** 2 + (x[i][1] - y[j][1]) ** 2)
    return z
