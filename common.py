import numpy as np
from scipy.spatial import distance_matrix
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


def get_distance_matrix(points):
    return distance_matrix(points, points)


def generate_points(dims=10, points=25):
    np.random.seed(1234)
    return np.random.rand(10, 25)


def get_numbers(dims=10, points=25, with_labels=False):
    np.random.seed(1234)
    result = load_digits(n_class=dims, return_X_y=True)
    # result = result if with_labels else result[0]
    return result[0][:points], result[1][:points] if with_labels else result[0][:points]


def cost_function(D, d, w=None, k=1, m=2):
    if w is None:
        w = 1 / D
    return np.sum(w * (D ** k - d ** k) ** m)


def plot(points, labels):
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], c=labels, label=labels)
    plt.show()
