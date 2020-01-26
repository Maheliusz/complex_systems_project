import numpy as np
from scipy.spatial import distance_matrix
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


def get_distance_matrix(points):
    return distance_matrix(points, points)


def generate_points(points=25, dims=10):
    np.random.seed(1234)
    return np.random.rand(points, dims)


def get_numbers(dims=10, points=25, with_labels=False):
    np.random.seed(1234)
    result = load_digits(n_class=dims, return_X_y=True)
    # result = result if with_labels else result[0]
    return result[0][:points], result[1][:points] if with_labels else result[0][:points]


def cost_function(D, d, w=None, k=2, m=0.5):#k=1, m=2):
    if w is None:
        # w = 1 #np.where(D!=0, 1.0/D, np.zeros(D.shape))
        w = np.ones(D.shape) - np.identity(D.shape[0])
    # return np.sum(w * (D ** k - d ** k) ** m)
    # stress = 0
    # for j in range(D.shape[0]):
    #     for i in range(j):
    #         stress += ((D[i, j]-d[i, j]) ** k) ** m
    # return np.sum((D - d) ** k) ** m
    # return stress
    return ((D.ravel() - d.ravel()) ** 2).sum() / 2


def plot(X, y, filename=None):
    _, plot = plt.subplots()
    plt.prism()
    for i in range(10):
        digit_indeces = y == i
        dim1 = X[digit_indeces, 0]
        dim2 = X[digit_indeces, 1]
        plot.scatter(dim1, dim2, label=str(i))
    plot.set_xticks(())
    plot.set_yticks(())
    plt.tight_layout()
    plt.legend()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

class MDSHolder:
    def __init__(self, initial_points, shape):
        self.initial_points = initial_points
        self.initial_distances = get_distance_matrix(self.initial_points)
        self.shape = shape

    def loss(self, points):
        distances = get_distance_matrix(points.reshape(self.shape))
        return cost_function(self.initial_distances, distances)