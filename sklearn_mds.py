from common import *
from sklearn.manifold import MDS

points_10d, labels = get_numbers(with_labels=True, points=500)
mds = MDS(verbose=10)
points_2d = mds.fit_transform(points_10d)
plot(points_2d, labels)
