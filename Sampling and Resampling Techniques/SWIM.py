# Import necessary libraries
import numpy as np  # For numerical computations and array manipulations
from sklearn.preprocessing import StandardScaler  # For standardizing data
import random  # For generating random numbers

# Custom exception for handling singular matrices
class SingularMatrixException(Exception):
    def __init__(self):
        Exception.__init__(self, "Singular data matrix... use subspace")

# Function to compute the square root of a symmetric matrix
def _msqrt(X):
    """
    Compute the square root matrix of a symmetric square matrix X.
    
    Parameters:
    - X: A symmetric square matrix.
    
    Returns:
    - The square root matrix of X.
    """
    (L, V) = np.linalg.eig(X)  # Eigen decomposition
    return V.dot(np.diag(np.sqrt(L))).dot(V.T)  # Reconstruct the square root matrix


# Class for SWIM-Maha oversampling
class SwimMaha:
    """
    SWIM-Maha (Synthetic Weighted Minority Oversampling with Mahalanobis Distance).
    
    Parameters:
    - sd: Standard deviation multiplier for generating synthetic samples.
    - minClass: The minority class label (optional).
    - subSpaceSampling: Whether to use subspace sampling for linearly dependent data (default is False).
    """
    def __init__(self, sd=0.25, minClass=None, subSpaceSampling=False):
        self.sd = sd  # Standard deviation multiplier
        self.minClass = minClass  # Minority class label
        self.subSpaceSampling = subSpaceSampling  # Subspace sampling flag

    # Method to perform Mahalanobis-based sampling
    def mahaSampling(self, data, labels, numSamples):
        """
        Perform Mahalanobis-based sampling to generate synthetic minority class samples.
        
        Parameters:
        - data: Feature matrix.
        - labels: Target labels.
        - numSamples: Number of synthetic samples to generate.
        
        Returns:
        - Oversampled feature matrix and labels.
        """
        # Determine the minority class if not provided
        if self.minClass is None:
            self.minClass = np.argmin(np.bincount(labels.astype(int)))

        syntheticInstances = []  # Store synthetic samples
        data_maj_orig = data[np.where(labels != self.minClass)[0], :]  # Majority class data
        data_min_orig = data[np.where(labels == self.minClass)[0], :]  # Minority class data

        # Reshape minority class data if it contains only one instance
        if np.sum(labels == self.minClass) == 1:
            data_min_orig = data_min_orig.reshape(1, len(data_min_orig))

        ## STEP 1: CENTRE THE DATA
        # Centre the majority class and minority class with respect to the majority class
        scaler = StandardScaler(with_std=False)  # Standardize data (mean=0, std=1)
        T_maj = np.transpose(scaler.fit_transform(data_maj_orig))  # Transpose majority class data
        T_min = np.transpose(scaler.transform(data_min_orig))  # Transpose minority class data

        ## STEP 2: WHITEN THE DATA
        C_inv = None
        C = np.cov(T_maj)  # Compute the covariance matrix of the majority class

        # Check the rank of the majority class data matrix
        data_rank = np.linalg.matrix_rank(data_maj_orig)
        if data_rank < T_maj.shape[0]:  # If there are linearly dependent columns
            if self.subSpaceSampling == False:
                print("The majority class has linearly dependent columns. Rerun the sampling with subSpaceSampling=True. Return original data.")
                return data, labels
            else:
                # Use QR decomposition to identify independent columns
                QR = np.linalg.qr(data_maj_orig)
                indep = QR[1].diagonal() > 0
                data = data[:, indep]
                print("The majority class has linearly dependent columns. Resampled data will be in the " + str(sum(indep == True)) + " independent columns of the original " + str(data_maj_orig.shape[1]) + "-dimensional data.")

        else:
            try:
                C_inv = np.linalg.inv(C)  # Invert the covariance matrix
            except np.linalg.LinAlgError as e:
                if 'Singular matrix' in str(e):
                    # Degrade to random oversampling with Gaussian jitter if the matrix is singular
                    print("Majority class data is singular. Degrading to random oversampling with Gaussian jitter")
                    X_new = data_min_orig[np.random.choice(data_min_orig.shape[0], numSamples, replace=True), :]
                    X_new = X_new + (0.1 * np.random.normal(0, data_maj_orig.std(0), X_new.shape))
                    y_new = np.repeat(self.minClass, numSamples)
                    data = np.concatenate([X_new, data])
                    labels = np.append(y_new, labels)
                    return data, labels

        try:
            # Compute the whitening transform matrix
            M = _msqrt(C_inv)  # Square root of the inverse covariance matrix
            M_inv = np.linalg.inv(M)  # Inverse of the whitening transform matrix

            # Apply the whitening transform to the minority and majority classes
            W_min = M.dot(T_min)  # Whitened minority class
            W_maj = M.dot(T_maj)  # Whitened majority class
        except:
            print("Value exception... synthetic instances not generated")
            return data, labels

        ## STEP 3: FIND THE MEANS AND FEATURE BOUNDS
        # Compute the mean and standard deviation of the whitened minority class
        min_means = W_min.mean(1)
        min_stds = W_min.std(1)
        min_ranges_bottom = min_means - self.sd * min_stds  # Lower bounds
        min_ranges_top = min_means + self.sd * min_stds  # Upper bounds

        ## STEP 4: GENERATE SYNTHETIC INSTANCES
        # Randomly replicate the whitened minority class instances to generate synthetic samples
        smpInitPts = W_min[:, np.random.choice(W_min.shape[1], numSamples)]
        for smpInd in range(smpInitPts.shape[1]):
            new_w_raw = []
            smp = smpInitPts[:, smpInd]
            for dim in range(len(min_means)):
                # Generate random values within the specified range
                new_w_raw.append(random.uniform(smp[dim] - self.sd * min_stds[dim], smp[dim] + self.sd * min_stds[dim]))

            ## STEP 5: SCALE BACK TO THE ORIGINAL SPACE
            # Normalize the synthetic sample and transform it back to the original space
            new_w = np.array(new_w_raw) / ((np.linalg.norm(new_w_raw) / np.linalg.norm(smp)))
            new = M_inv.dot(np.array(new_w))
            syntheticInstances.append(new)

        # Convert synthetic instances to a real-valued array
        syntheticInstances = np.real(syntheticInstances)

        # Combine synthetic samples with the original data
        sampled_data = np.concatenate([scaler.inverse_transform(np.array(syntheticInstances)), data])
        sampled_labels = np.append([self.minClass] * len(syntheticInstances), labels)

        return sampled_data, sampled_labels