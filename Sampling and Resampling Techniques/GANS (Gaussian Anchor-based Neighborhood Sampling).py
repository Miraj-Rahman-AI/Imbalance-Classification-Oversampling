import numpy as np
import random
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from collections import Counter

class GDO:
    def __init__(self, k=5, alpha=1):
        """
        Initialize the GDO (Gaussian Distribution Oversampling) class.

        Parameters:
        -----------
        k : int, optional (default=5)
            The number of neighbors to consider for each instance.
        alpha : float, optional (default=1)
            The covariance coefficient used in the Gaussian distribution for generating new instances.
        """
        self.k = k  # Number of neighbors
        self.alpha = alpha  # Covariance coefficient
        self.N = 0  # Number of minority samples
        self.M = 0  # Number of majority samples
        self.l = 0  # Dimension of input data
        self.min_index = []  # Index of minority samples
        self.maj_index = []  # Index of majority samples

    def normalize(self, a):
        """
        Normalize the input array to the range [0, 1].

        Parameters:
        -----------
        a : numpy array
            The input array to be normalized.

        Returns:
        --------
        numpy array
            Normalized array.
        """
        a = a.reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()  # Normalize to 0-1
        a = min_max_scaler.fit_transform(a)
        return a.reshape(1, -1)[0]

    def Minority_instance_weighting(self, X, y, dist, indices):
        """
        Calculate the information weight for each minority instance.

        Parameters:
        -----------
        X : numpy array
            The input features.
        y : numpy array
            The target labels.
        dist : numpy array
            The distances to the k-nearest neighbors for each instance.
        indices : numpy array
            The indices of the k-nearest neighbors for each instance.

        Returns:
        --------
        numpy array
            Normalized information weights for minority instances.
        """
        C = np.zeros(self.N)  # Density factor
        D = np.zeros(self.N)  # Distance factor
        I = np.zeros(self.N)  # Information weight

        for i, index in enumerate(self.min_index[0]):
            neigh_label = y[indices[index, 1:self.k + 1]]
            K_Ni_maj = Counter(neigh_label)[1]
            C[i] = K_Ni_maj / self.k

            neigh_maj_index = np.where(neigh_label == 1)[0] + 1  # Index of majority neighbors
            dist_to_NN_all = sum(dist[index])  # Distance from Xi to all neighbors
            dist_to_NN_maj = sum(dist[index, neigh_maj_index])  # Distance from Xi to majority neighbors
            D[i] = dist_to_NN_maj / dist_to_NN_all

        I = C + D  # Information weight

        return self.normalize(I)

    def Probabilistic_anchor_instance_selection(self, I):
        """
        Select an anchor instance from the minority class based on the information weights.

        Parameters:
        -----------
        I : numpy array
            The information weights for minority instances.

        Returns:
        --------
        int
            The index of the selected anchor instance.
        """
        a = [i for i in range(self.N)]  # Indices of minority instances
        gamma = random.choices(a, weights=I, k=1)[0]  # Roulette selection
        return gamma

    def New_instance_generation(self, I, min_sample):
        """
        Generate new synthetic instances for the minority class.

        Parameters:
        -----------
        I : numpy array
            The information weights for minority instances.
        min_sample : numpy array
            The minority class samples.

        Returns:
        --------
        numpy array
            The generated synthetic instances.
        """
        k = 1
        G = self.M - self.N  # Number of samples to generate
        new_instances = []

        neigh = NearestNeighbors(n_neighbors=2).fit(min_sample)
        dist_min, indices_min = neigh.kneighbors(min_sample)

        while k <= G:
            selected_index = self.Probabilistic_anchor_instance_selection(I)
            anchor = min_sample[selected_index]
            V = np.random.uniform(-1, 1, size=(1, self.l))[0]  # Random direction vector

            d_0 = np.linalg.norm(anchor - V)
            mu = 0
            sigma = dist_min[selected_index, 1]  # Distance to the nearest minority neighbor
            d_i = self.alpha * sigma * np.random.randn(1) + mu  # Random distance based on Gaussian distribution
            r = d_i / d_0

            synthetic_instance = anchor + r * (V - anchor)
            new_instances.append(synthetic_instance)

            k += 1
        return np.array(new_instances)

    def fit_sample(self, X, y):
        """
        Fit the model and generate synthetic samples to balance the dataset.

        Parameters:
        -----------
        X : numpy array
            The input features.
        y : numpy array
            The target labels.

        Returns:
        --------
        tuple
            A tuple containing the resampled features and labels.
        """
        self.min_index = np.where(y == 1)  # Indices of minority samples
        self.maj_index = np.where(y == 0)  # Indices of majority samples

        min_sample = X[self.min_index]
        maj_sample = X[self.maj_index]

        self.N = len(min_sample)
        self.M = len(maj_sample)
        self.l = X.shape[1]

        neigh = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        dist, indices = neigh.kneighbors(X)

        I = self.Minority_instance_weighting(X, y, dist, indices)

        new_instances = self.New_instance_generation(I, min_sample)
        new_y = np.array([1] * len(new_instances))

        Resampled_X = np.concatenate((X, new_instances), axis=0)
        Resampled_y = np.concatenate((y, new_y), axis=0)

        return Resampled_X, Resampled_y

# Example usage:
# X, y = GDO(k=5, alpha=1).fit_sample(X, y)