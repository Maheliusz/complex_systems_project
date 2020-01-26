from common import *
import random
from scipy.optimize import minimize
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer
import numpy as np

max_iter = 3000

points_md, labels = get_numbers(with_labels=True, points=500)
n_samples = points_md.shape[0]
# print(points_md.shape)
points_2d = generate_points(n_samples, 2)
X = np.copy(points_2d)
disparities = get_distance_matrix(points_md)

old_stress = None
V = np.zeros(points_2d.shape)
M = np.zeros(points_2d.shape)
stress = 0
for it in range(max_iter):
    break
    # Compute distance and monotonic regression
    dis = get_distance_matrix(X)

    # Compute stress
    old_stress = stress
    stress = ((dis.ravel() - disparities.ravel()) ** 2).sum() / 2
    print('it: %d, stress %s' % (it, stress), end='\r')
    if old_stress < stress and it > max_iter/10:
        print()
        plot(X, labels, 'nadam.png')
        break

    # Update X using the Guttman transform
    dis[dis == 0] = 1e-5
    ratio = disparities / dis
    B = - ratio
    B[np.arange(len(B)), np.arange(len(B))] += ratio.sum(axis=1)
    X = 1. / n_samples * np.dot(B, X)
else:
    print()
    plot(X, labels, 'guttman.png')

# X = np.copy(points_2d)
m = np.zeros(X.shape)
v = np.zeros(X.shape)
epsilon = 1e-8
step_size = 0.001
beta_1 = 0.9
beta_2 = 0.999
stress = 0
for it in range(1, max_iter+1):
    # break
    # Compute distance and monotonic regression
    dis = get_distance_matrix(X)

    # Compute stress
    old_stress = stress
    stress = ((dis.ravel() - disparities.ravel()) ** 2).sum() / 2
    print('it: %d, stress %s' % (it, stress), end='\r')
    if old_stress < stress and it > max_iter/10:
        print()
        plot(X, labels, 'nadam.png')
        break

    # X -= dis
    # np.random.shuffle(dis)
    for i in range(dis.shape[1]):
        # g = dis[:, np.random.choice(dis.shape[1], 2, replace=False)]
        g = dis[:, i:i+1]

        m = beta_1 * m + (1 - beta_1) * g
        v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
        m_hat = m / (1 - np.power(beta_1, it)) + (1 - beta_1) * g / (1 - np.power(beta_1, it))
        v_hat = v / (1 - np.power(beta_2, it))
        X -= step_size * m_hat / (np.sqrt(v_hat) + epsilon)        
else:
    print()
    plot(X, labels, 'nadam.png')