# Calculate distances matrix between geographical locations
# Reference: https://en.wikipedia.org/wiki/Haversine_formula
import numpy as np


def hav(theta):
    return np.sin(theta / 2.) ** 2


def earthDistances(rowloc=None, colloc=None):
    """
    :param rowloc: a N*2 array representing (latitude, longitude) of N locations, dtype='float64'
    :param colloc: a M*2 array representing (latitude, longitude) of N locations, dtype='float64'
    :return: a matrix of distances between locations
    """
    radius = 6378.137  # Radius of Earth in km

    if not colloc:
        colloc = rowloc.copy()

    rowloc = rowloc * np.pi / 180.
    colloc = colloc * np.pi / 180.
    distances = np.zeros((rowloc.shape[0], colloc.shape[0]))
    for i in range(colloc.shape[0]):
        distances[:, i] = radius * 2 * np.arcsin(np.sqrt(hav(rowloc[:, 0] - colloc[i, 0]) + (
                1 - hav(rowloc[:, 0] - colloc[i, 0]) - hav(rowloc[:, 0] + colloc[i, 0])) * hav(
            rowloc[:, 1] - colloc[i, 1])))
    return distances
