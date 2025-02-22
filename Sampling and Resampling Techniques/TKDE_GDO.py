# Import necessary libraries
import numpy as np  # For numerical operations
import random  # For random number generation
from sklearn import preprocessing  # For data normalization
from sklearn.neighbors import NearestNeighbors  # For finding nearest neighbors
from collections import Counter  # For counting occurrences of elements

class GDO:
    def __init__(self, k=5, alpha=1):
        """
        Initialize the GDO (Geometric Direction Oversampling) class.

        Parameters:
        -----------
        k : int, optional (default=5)
            The number of nearest neighbors to consider.
        alpha : float, optional (default=1)
            The covariance coefficient used in the Gaussian distribution for generating new instances.
        """
        self.k = k  # Number of nearest neighbors
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
            The normalized array.
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
            The distances to the nearest neighbors.
        indices : numpy array
            The indices of the nearest neighbors.

        Returns:
        --------
        numpy array
            The normalized information weights for minority instances.
        """
        C = np.zeros(self.N)  # Density factor
        D = np.zeros(self.N)  # Distance factor
        I = np.zeros(self.N)  # Information weight

        for i, index in enumerate(self.min_index[0]):
            # Get the labels of the k-nearest neighbors
            neigh_label = y[indices[index, 1:self.k + 1]]
            # Count the number of majority class instances in the neighborhood
            K_Ni_maj = Counter(neigh_label)[1]
            C[i] = K_Ni_maj / self.k

            # Get the indices of majority class neighbors
            neigh_maj_index = np.where(neigh_label == 1)[0] + 1
            # Calculate the total distance to all neighbors
            dist_to_NN_all = sum(dist[index])
            # Calculate the total distance to majority class neighbors
            dist_to_NN_maj = sum(dist[index, neigh_maj_index])
            D[i] = dist_to_NN_maj / dist_to_NN_all

        # Information weight is the sum of density and distance factors
        I = C + D

        return self.normalize(I)

    def Probabilistic_anchor_instance_selection(self, I):
        """
        Select an anchor instance from the minority class using roulette wheel selection.

        Parameters:
        -----------
        I : numpy array
            The information weights of minority instances.

        Returns:
        --------
        int
            The index of the selected anchor instance.
        """
        a = [i for i in range(self.N)]  # List of minority instance indices
        # Roulette wheel selection based on information weights
        gamma = random.choices(a, weights=I, k=1)[0]

        return gamma

    def New_instance_generation(self, I, min_sample):
        """
        Generate new synthetic instances for the minority class.

        Parameters:
        -----------
        I : numpy array
            The information weights of minority instances.
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

        # Find the nearest neighbors of minority samples
        neigh = NearestNeighbors(n_neighbors=2).fit(min_sample)
        dist_min, indices_min = neigh.kneighbors(min_sample)

        while k <= G:
            # Select an anchor instance
            selected_index = self.Probabilistic_anchor_instance_selection(I)
            anchor = min_sample[selected_index]
            # Randomly select a direction originating from the anchor instance
            V = np.random.uniform(-1, 1, size=(1, self.l))[0]

            # Calculate the distance between the anchor and the random direction
            d_0 = np.linalg.norm(anchor - V)
            mu = 0
            # The distance between the anchor and its nearest minority neighbor
            sigma = dist_min[selected_index, 1]
            # Generate a random distance based on Gaussian distribution
            d_i = self.alpha * sigma * np.random.randn(1) + mu
            r = d_i / d_0

            # Generate a synthetic instance
            synthetic_instance = anchor + r * (V - anchor)
            new_instances.append(synthetic_instance)

            k += 1
        return np.array(new_instances)

    def fit_sample(self, X, y):
        """
        Fit the model and generate synthetic samples.

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
        # Identify minority and majority class indices
        self.min_index = np.where(y == 1)  # Minority class label is 1
        self.maj_index = np.where(y == 0)  # Majority class label is 0

        min_sample = X[self.min_index]
        maj_sample = X[self.maj_index]

        self.N = len(min_sample)  # Number of minority samples
        self.M = len(maj_sample)  # Number of majority samples
        self.l = X.shape[1]  # Dimension of input data

        # Find the k-nearest neighbors for each instance
        neigh = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        dist, indices = neigh.kneighbors(X)

        # Calculate information weights for minority instances
        I = self.Minority_instance_weighting(X, y, dist, indices)

        # Generate new synthetic instances
        new_instances = self.New_instance_generation(I, min_sample)
        new_y = np.array([1] * len(new_instances))  # Labels for new instances

        # Combine original and synthetic instances
        Resampled_X = np.concatenate((X, new_instances), axis=0)
        Resampled_y = np.concatenate((y, new_y), axis=0)

        return Resampled_X, Resampled_y


# Example usage:
# X, y = GDO(k=5, alpha=1).fit_sample(X, y)