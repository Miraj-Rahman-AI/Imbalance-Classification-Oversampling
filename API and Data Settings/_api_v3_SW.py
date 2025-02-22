# Import necessary libraries
import warnings  # For handling warnings
from collections import OrderedDict  # For ordered dictionaries
from functools import wraps  # For function decorators
from inspect import signature, Parameter  # For inspecting function signatures
from numbers import Integral, Real  # For checking numeric types
import numpy as np  # For numerical operations
from sklearn.base import clone  # For cloning sklearn estimators
from sklearn.neighbors._base import KNeighborsMixin  # Base class for k-NN estimators
from sklearn.neighbors import NearestNeighbors  # For k-NN computations
from sklearn.utils import column_or_1d  # For ensuring column vectors
from sklearn.utils.multiclass import type_of_target  # For checking target types
from imblearn.exceptions import raise_isinstance_error  # For raising type errors
from sklearn.utils.fixes import np_version, parse_version  # For handling version-specific fixes
from scipy.sparse import issparse  # For checking sparse matrices
from sklearn.utils import _safe_indexing  # For safe indexing of arrays
from itertools import compress  # For filtering using boolean masks
import random  # For generating random numbers

# Suppress warnings to avoid clutter in the output
warnings.filterwarnings('ignore')

# Define a function to check if an object is consistent to be a k-NN object
def check_neighbors_object(nn_name, nn_object, additional_neighbor=0):
    """
    Check if the object is consistent to be a k-NN object.

    Parameters:
    ----------
    nn_name : str
        The name associated with the object to raise an error if needed.

    nn_object : int or KNeighborsMixin
        The object to be checked. It can be an integer (number of neighbors) or a k-NN object.

    additional_neighbor : int, optional (default=0)
        Sometimes, some algorithms need an additional neighbor.

    Returns:
    -------
    nn_object : KNeighborsMixin
        The k-NN object.

    Raises:
    ------
    TypeError
        If the object is neither an integer nor a KNeighborsMixin.
    """
    if isinstance(nn_object, Integral):
        # If the object is an integer, create a NearestNeighbors object with the specified number of neighbors
        return NearestNeighbors(n_neighbors=nn_object + additional_neighbor)
    elif isinstance(nn_object, KNeighborsMixin):
        # If the object is already a k-NN object, clone it
        return clone(nn_object)
    else:
        # Raise an error if the object is neither an integer nor a k-NN object
        raise_isinstance_error(nn_name, [int, KNeighborsMixin], nn_object)

# Define a function to identify samples in danger or noise
def in_danger_noise(nn_estimator, samples, target_class, y, kind="danger"):
    """
    Identify samples that are in danger or noise based on their k-NN neighbors.

    Parameters:
    ----------
    nn_estimator : KNeighborsMixin
        The k-NN estimator used to find neighbors.

    samples : array-like
        The samples to be checked.

    target_class : int
        The target class label.

    y : array-like
        The target labels of the dataset.

    kind : str, optional (default="danger")
        The type of samples to identify. Can be "danger" or "noise".

    Returns:
    -------
    mask : ndarray
        A boolean mask indicating which samples are in danger or noise.

    n_maj : ndarray
        The number of majority class samples in the k-NN neighborhood of each sample.

    Raises:
    ------
    NotImplementedError
        If the kind is not "danger" or "noise".
    """
    # Find the k-NN neighbors of the samples (excluding the sample itself)
    x = nn_estimator.kneighbors(samples, return_distance=False)[:, 1:]
    # Check if the neighbors belong to the target class
    nn_label = (y[x] != target_class).astype(int)
    # Count the number of majority class samples in the neighborhood
    n_maj = np.sum(nn_label, axis=1)

    if kind == "danger":
        # Samples are in danger if the number of majority class samples is between (m/2) and m
        return np.bitwise_and(
            n_maj >= (nn_estimator.n_neighbors - 1) / 2,
            n_maj < nn_estimator.n_neighbors - 1,
        ), n_maj
    elif kind == "noise":
        # Samples are noise if all neighbors are majority class samples
        return n_maj == nn_estimator.n_neighbors - 1, n_maj
    else:
        # Raise an error if the kind is not recognized
        raise NotImplementedError

# Define a function to add weights to samples based on their neighborhood
def add_weight(X, y, X_min, minority_label, base_indices, neighbor_indices, num_to_sample, ind, X_neighbor, X_base, weight, ntree):
    """
    Add weights to samples based on their neighborhood.

    Parameters:
    ----------
    X : array-like
        The feature matrix of the dataset.

    y : array-like
        The target labels of the dataset.

    X_min : array-like
        The minority class samples.

    minority_label : int
        The label of the minority class.

    base_indices : array-like
        The indices of the base samples.

    neighbor_indices : array-like
        The indices of the neighbor samples.

    num_to_sample : int
        The number of samples to generate.

    ind : array-like
        The indices of the neighbors for each base sample.

    X_neighbor : array-like
        The neighbor samples.

    X_base : array-like
        The base samples.

    weight : array-like
        The weights of the samples.

    ntree : int
        The number of trees used in the weight calculation.

    Returns:
    -------
    samples : ndarray
        The generated samples with weights applied.
    """
    # Extract weights for the minority class samples
    weight_maj = _safe_indexing(weight, np.flatnonzero(y == minority_label))
    # Compute new weights based on the number of trees
    new_n_maj = np.array([round((1 - i / ntree), 2) for i in weight_maj])

    # Extract weights for the base and neighbor samples
    X_base_weight = new_n_maj[base_indices]  # Weights for the base samples
    X_neighbor_weight = new_n_maj[ind[base_indices, neighbor_indices]]  # Weights for the neighbor samples

    # Initialize lists to store weights and indices of samples to delete
    weights = []
    delete_index = []

    # Iterate over the number of samples to generate
    for n in range(int(num_to_sample)):
        if X_base_weight[n] != 0 and X_neighbor_weight[n] != 0:
            # If both base and neighbor samples are not noise
            if X_base_weight[n] >= X_neighbor_weight[n]:
                # Compute the proportion based on the weights
                proportion = (X_neighbor_weight[n] / (X_base_weight[n] + X_neighbor_weight[n]) * round(random.uniform(0, 1), len(str(num_to_sample))))
            elif X_base_weight[n] < X_neighbor_weight[n]:
                proportion = X_neighbor_weight[n] / (X_base_weight[n] + X_neighbor_weight[n])
                proportion = proportion + (1 - proportion) * (round(random.uniform(0, 1), len(str(num_to_sample))))
        
        # Append the computed proportion to the weights list
        weights.append(proportion)

    # Delete samples marked for deletion
    X_neighbor = np.delete(X_neighbor, delete_index, axis=0)
    X_base = np.delete(X_base, delete_index, axis=0)

    # Reshape the weights array
    weights = np.array(weights).reshape(int(len(weights)), 1)

    # Generate new samples by applying the weights
    samples = X_base + np.multiply(weights, X_neighbor - X_base)
    return samples