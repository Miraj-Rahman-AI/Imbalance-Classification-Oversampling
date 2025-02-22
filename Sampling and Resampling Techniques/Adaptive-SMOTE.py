import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter

class ASmote:
    """
    Adaptive Synthetic Minority Over-sampling Technique (ASMOTE) is a class designed to handle imbalanced datasets
    by generating synthetic samples for the minority class. The class provides methods to oversample the minority class
    to balance the dataset.

    Usage:
    data_Asmote = ASmote(data)
    data_Asmote.over_sampling()
    """

    def __init__(self, data, random_state=None):
        """
        Initializes the ASmote class with the dataset.

        :param data: numpy.ndarray, the dataset with labels in the 0th column.
        :param random_state: int, optional, seed for random number generation for reproducibility.
        """
        self.data = data  # The dataset with labels in the first column

    def dataReset(self):
        """
        Splits the dataset into majority and minority classes based on the labels.

        Returns:
        - X: numpy.ndarray, the entire dataset without labels.
        - Xn: numpy.ndarray, the majority class data without labels.
        - tp_more: int, the label of the majority class.
        - Xp: numpy.ndarray, the minority class data without labels.
        - tp_less: int, the label of the minority class.
        """
        # Count the occurrences of each class label
        count = Counter(self.data[:, 0])
        if len(count) != 2:
            raise Exception('Dataset must have exactly 2 classes. Found: {}'.format(count))
        
        # Identify majority and minority classes
        tp_more, tp_less = set(count.keys())
        if count[tp_more] < count[tp_less]:
            tp_more, tp_less = tp_less, tp_more
        
        # Split the data into majority and minority classes
        data_more = self.data[self.data[:, 0] == tp_more]
        data_less = self.data[self.data[:, 0] == tp_less]
        
        # Extract features (excluding labels)
        X = self.data[:, 1:]
        Xn = data_more[:, 1:]
        Xp = data_less[:, 1:]

        return X, Xn, tp_more, Xp, tp_less

def oversampleASmote(data, N, Xp):
    """
    Oversamples the minority class using the ASMOTE algorithm.

    :param data: numpy.ndarray, the entire dataset (majority + minority classes).
    :param N: numpy.ndarray, the majority class data.
    :param Xp: numpy.ndarray, the minority class data.
    :return: numpy.ndarray, the oversampled minority class data.
    """
    Nnum = N.shape[0]  # Number of majority class samples
    Pnum = Xp.shape[0]  # Number of minority class samples
    dim = N.shape[1]  # Number of features

    if Pnum <= 1.05 * Nnum:
        # Identify boundary and inner points in the minority class
        boundary, inner = boundary_inner(data, Xp, Nnum, dim)
        
        if len(boundary) > 1 and len(inner) > 1:
            # Generate new synthetic samples
            new = new1_data(data, Xp, inner, boundary, Nnum, dim)[1:]
        elif Pnum == 0:
            # If no minority samples, perform random SMOTE
            new = random_SMOTE(Xp, int(round(Nnum)))
        else:
            # Perform random SMOTE based on the ratio
            new = random_SMOTE(Xp, int(round(Nnum / Pnum)))
        
        # Append new synthetic samples to the minority class
        Xp = np.vstack((Xp, new))
        data = np.vstack((data, new))
        
        if len(Xp) < Nnum:
            if len(Xp) == 0:
                new1 = random_SMOTE(Xp, int(round(Nnum)))
            else:
                new1 = random_SMOTE(Xp, int(round(Nnum / len(Xp))))
            Xp = np.vstack((Xp, new1))
            data = np.vstack((data, new1))
    
    return Xp

def boundary_inner(X, X_1, Nnum, b):
    """
    Identifies boundary and inner points in the minority class.

    :param X: numpy.ndarray, the entire dataset.
    :param X_1: numpy.ndarray, the minority class data.
    :param Nnum: int, the number of majority class samples.
    :param b: int, the number of features.
    :return: tuple, (boundary points, inner points).
    """
    inner0 = np.zeros((1, b + 1))
    boundary0 = np.zeros((1, b + 1))
    r, c = X_1.shape
    
    for i in range(r):
        inner = False
        for k in range(5, 11):
            if np.sum(neigh0(X, X_1, k, i) >= Nnum) > k / 2:
                inner = True
        if inner:
            inner0 = np.vstack((inner0, np.hstack((i, X_1[i]))))
        else:
            boundary0 = np.vstack((boundary0, np.hstack((i, X_1[i]))))
    
    return boundary0, inner0

def new1_data(data, P, inner, boundary, Nnum, b):
    """
    Generates new synthetic samples for the minority class.

    :param data: numpy.ndarray, the entire dataset.
    :param P: numpy.ndarray, the minority class data.
    :param inner: numpy.ndarray, the inner points of the minority class.
    :param boundary: numpy.ndarray, the boundary points of the minority class.
    :param Nnum: int, the number of majority class samples.
    :param b: int, the number of features.
    :return: numpy.ndarray, the new synthetic samples.
    """
    Pnew = np.zeros((1, b))
    for i in range(1, len(inner)):
        [nn] = neigh0(data, inner[:, 1:], 10, i)
        AND = set(list(nn - Nnum)).intersection(set(list(boundary[1:, 0])))
        AND = list(AND)
        for j in range(len(AND)):
            AND[j] = int(AND[j])
        if not (AND == set([])):
            for j in AND:
                dif = P[j] - inner[i, 1:]
                Pnew = np.vstack((Pnew, inner[i, 1:] + np.random.rand() * dif))
    return Pnew

def neigh0(data, P, k, i):
    """
    Finds the k-nearest neighbors of a point in the minority class.

    :param data: numpy.ndarray, the entire dataset.
    :param P: numpy.ndarray, the minority class data.
    :param k: int, the number of nearest neighbors.
    :param i: int, the index of the point in the minority class.
    :return: numpy.ndarray, the indices of the k-nearest neighbors.
    """
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(data)
    nn = neigh.kneighbors(P[i].reshape(1, -1), return_distance=False)
    return nn

def random_SMOTE(data, N):
    """
    Generates synthetic samples using the random SMOTE technique.

    :param data: numpy.ndarray, the minority class data.
    :param N: int, the number of synthetic samples to generate.
    :return: numpy.ndarray, the new synthetic samples.
    """
    sample_num, feature_dim = data.shape
    new_data = np.zeros((sample_num * N, feature_dim))
    tmp_data = np.zeros((N, feature_dim))

    for i in range(sample_num):
        X = data[i]
        idx1 = _rand_idx(0, sample_num - 1, (i,))
        idx2 = _rand_idx(0, sample_num - 1, (i, idx1))
        Y1 = data[idx1]
        Y2 = data[idx2]

        for j in range(N):
            for k in range(feature_dim):
                dif = Y2[k] - Y1[k]
                tmp_data[j][k] = Y1[k] + dif * np.random.rand()

        for j in range(N):
            for k in range(feature_dim):
                dif = tmp_data[j][k] - X[k]
                new_data[i * N + j][k] = X[k] + dif * np.random.rand()

    return new_data

def _rand_idx(start, end, exclude=None):
    """
    Generates a random index within a range, excluding specified indices.

    :param start: int, the start of the range.
    :param end: int, the end of the range.
    :param exclude: tuple, indices to exclude.
    :return: int, a random index within the range.
    """
    if start > end:
        start, end = end, start
    elif start == end:
        return start
    
    rev = np.random.randint(start, end)
    if exclude is not None:
        while rev in exclude:
            rev = np.random.randint(start, end)
    return rev