# from sklearn.neighbors import NearestNeighbors
from io import SEEK_CUR
from typing import Type
from numpy.core.defchararray import partition
from numpy.lib.type_check import real
from sklearn.neighbors import KDTree
# import sklearn.neighbors
import pandas as pd
import numpy as np
import math
import csv
from matplotlib import pyplot as plt
# from KBsomte.SMOTE.Density import RelativeDensity
# from experiment_utils import load_2classes_data
from collections import Counter

class Natural_Neighbor:
    """
    A class to compute Natural Neighbors and Relative Density for a given dataset.
    Natural Neighbors are mutual nearest neighbors, and Relative Density is a measure
    of how dense a point is relative to its neighbors.
    """

    def __init__(self):
        """
        Initialize the Natural_Neighbor class with empty structures.
        """
        self.nan_edges = {}  # Graph of mutual neighbors
        self.nan_num = {}   # Number of natural neighbors of each instance
        self.target = []     # Set of class labels
        self.data = []       # Instance set (features)
        self.knn = {}        # Structure that stores the neighbors of each instance
        self.nan = {}        # Structure that stores the natural neighbors of each instance
        self.relative_cox = []  # Relative density values for each instance

    def asserts(self):
        """
        Initialize the data structures required for computing natural neighbors.
        """
        self.nan_edges = set()  # Initialize the set of mutual neighbor edges
        for j in range(len(self.data)):
            self.knn[j] = set()  # Initialize the k-nearest neighbors set for each instance
            self.nan[j] = set()   # Initialize the natural neighbors set for each instance
            self.nan_num[j] = 0   # Initialize the count of natural neighbors to zero

    def count(self):
        """
        Count the number of instances that have no natural neighbors.
        """
        nan_zeros = 0
        for x in self.nan_num:
            if self.nan_num[x] == 0:
                nan_zeros += 1
        return nan_zeros

    def findKNN(self, inst, r, tree):
        """
        Find the k-nearest neighbors of a given instance using a KDTree.
        
        Parameters:
        - inst: The instance for which to find neighbors.
        - r: The number of neighbors to find.
        - tree: The KDTree used for efficient neighbor search.
        
        Returns:
        - The indices of the closest neighbors.
        """
        dist, ind = tree.query([inst], r+1)  # Query the tree for r+1 nearest neighbors
        return np.delete(ind[0], 0)  # Remove the first element (the instance itself)

    def algorithm(self):
        """
        Compute the natural neighbors for each instance in the dataset.
        
        Returns:
        - The number of neighbors (r) used to compute natural neighbors.
        """
        tree = KDTree(self.data)  # Build a KDTree for efficient neighbor search
        self.asserts()  # Initialize the data structures
        flag = 0
        r = 2  # Start with r=2 (natural feature initialized to 1)
        cnt_before = -1  # Initialize the count of instances with zero natural neighbors

        while flag == 0:
            for i in range(len(self.data)):
                knn = self.findKNN(self.data[i], r, tree)  # Find the k-nearest neighbors
                n = knn[-1]  # Get the r-th nearest neighbor
                self.knn[i].add(n)  # Add the neighbor to the instance's neighbor set
                if i in self.knn[n] and (i, n) not in self.nan_edges:  # Check if mutual neighbors
                    self.nan_edges.add((i, n))  # Add the mutual neighbor edge
                    self.nan_edges.add((n, i))
                    self.nan[i].add(n)  # Add to natural neighbors
                    self.nan[n].add(i)
                    self.nan_num[i] += 1  # Increment the natural neighbor count
                    self.nan_num[n] += 1

            cnt_after = self.count()  # Count instances with zero natural neighbors
            if cnt_after < math.sqrt(len(self.data)):  # Stop if the count is below a threshold
                flag = 1
            else:
                r += 1  # Increase the number of neighbors to consider
            cnt_before = cnt_after
        return r

    def RelativeDensity(self, min_i, maj_i):
        """
        Compute the relative density for each instance.
        
        Parameters:
        - min_i: The minority class label.
        - maj_i: The majority class label.
        """
        self.relative_cox = [0] * len(self.target)  # Initialize relative density values
        for i, num in self.nan.items():  # num is the set of natural neighbors for instance i
            if self.target[i] == min_i:  # Only compute for minority class instances
                if len(num) == 0:  # Mark as outlier if no natural neighbors
                    self.relative_cox[i] = -2
                else:
                    absolute_min, min_num, absolute_max, maj_num = 0, 0, 0, 0
                    maj_index = []

                    for j in iter(num):  # Iterate over natural neighbors
                        if self.target[j] == min_i:
                            absolute_min += np.sqrt(np.sum(np.square(self.data[i] - self.data[j])))
                            min_num += 1
                        elif self.target[j] == maj_i:
                            absolute_max += np.sqrt(np.sum(np.square(self.data[i] - self.data[j])))
                            maj_num += 1
                            maj_index.append(j)
                    self.nan[i].difference_update(maj_index)  # Remove majority class neighbors

                    if min_num == 0:  # Mark as noise if all neighbors are majority class
                        self.relative_cox[i] = -3
                    elif maj_num == 0:  # Safe point if all neighbors are minority class
                        relative = min_num / absolute_min
                        self.relative_cox[i] = relative
                    else:  # Borderline point if neighbors are mixed
                        relative = (min_num / absolute_min) / (maj_num / absolute_max)
                        self.relative_cox[i] = relative

    def RelativeDensity_7(self, min_i, maj_i):
        """
        Compute the relative density with a threshold of 0.7 for noise detection.
        
        Parameters:
        - min_i: The minority class label.
        - maj_i: The majority class label.
        """
        self.relative_cox = [0] * len(self.target)
        for i, num in self.nan.items():
            if self.target[i] == min_i:
                if len(num) == 0:
                    self.relative_cox[i] = -2
                else:
                    absolute_min, min_num, absolute_max, maj_num = 0, 0, 0, 0
                    maj_index = []

                    for j in iter(num):
                        if self.target[j] == min_i:
                            absolute_min += np.sqrt(np.sum(np.square(self.data[i] - self.data[j])))
                            min_num += 1
                        elif self.target[j] == maj_i:
                            absolute_max += np.sqrt(np.sum(np.square(self.data[i] - self.data[j])))
                            maj_num += 1
                            maj_index.append(j)
                    self.nan[i].difference_update(maj_index)

                    if min_num == 0 or maj_num >= (min_num + maj_num) * 0.7:  # Noise if majority neighbors >= 70%
                        self.relative_cox[i] = -3
                    elif maj_num == 0:
                        relative = min_num / absolute_min
                        self.relative_cox[i] = relative
                    else:
                        relative = (min_num / absolute_min) / (maj_num / absolute_max)
                        self.relative_cox[i] = relative

def NaN_RD(data_train):
    """
    Compute Natural Neighbors and Relative Density for a given dataset.
    
    Parameters:
    - data_train: The dataset with features and labels.
    
    Returns:
    - data: The original dataset with labels and features.
    - relative_cox: The relative density values for each instance.
    - nan: The natural neighbors for each instance.
    """
    NN = Natural_Neighbor()
    NN.data = data_train[:, 1:]  # Initialize the dataset (features)
    NN.target = data_train[:, 0]  # Initialize the labels
    NN.algorithm()  # Compute natural neighbors

    count = Counter(NN.target)  # Get the minority and majority class labels
    c = count.most_common(len(count))
    min_i, maj_i = c[1][0], c[0][0]

    # Compute relative density
    NN.RelativeDensity_7(min_i, maj_i)
    NN.relative_cox = np.array(NN.relative_cox)  # Convert to numpy array

    # Combine data features and labels
    data = np.c_[NN.target, NN.data]

    # Convert natural neighbors to a numpy array
    NN.nan = np.array([list(v) for v in NN.nan.values()])

    return data, NN.relative_cox, NN.nan  # Return the original data, relative density, and natural neighbors