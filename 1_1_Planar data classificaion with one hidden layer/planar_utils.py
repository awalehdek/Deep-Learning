import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def load_planar_dataset():
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()



def sigmoid(z):
    """
    Computes the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Returns:
    s -- sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))
    return s


from sklearn import datasets


def load_extra_datasets():
    """
    Loads additional datasets for testing classification algorithms.

    Returns:
    datasets -- a dictionary containing various datasets ('moons', 'circles', 'blobs', and 'gaussian_quantiles')
    """
    np.random.seed(0)
    N = 200  # number of points per class

    # Moons dataset
    noisy_moons = datasets.make_moons(n_samples=N, noise=0.2)

    # Circles dataset
    noisy_circles = datasets.make_circles(n_samples=N, factor=0.5, noise=0.3)

    # Blobs dataset
    blobs = datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=2, cluster_std=1.0)

    # Gaussian Quantiles
    gaussian_quantiles = datasets.make_gaussian_quantiles(mean=None, cov=0.9, n_samples=N, n_features=2, n_classes=2)

    # Dictionary of datasets
    datasets = {"noisy_moons": noisy_moons,
                "noisy_circles": noisy_circles,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}

    return datasets
