# Import necessary libraries
import numpy as np  # For numerical operations
import random  # For generating random numbers

# Define a function to add weights to samples based on their neighborhood
def add_weight(X_neighbor, X_base, X_base_weight, X_neighbor_weight):
    """
    Add weights to samples based on their neighborhood to generate new synthetic samples.

    This function calculates the interpolation weights between base samples and their neighbors
    to generate new synthetic samples. The weights are computed based on the relative importance
    of the base and neighbor samples, and a random factor is introduced to add variability.

    Parameters:
    ----------
    X_neighbor : array-like
        The neighbor samples. These are the samples that are close to the base samples in the feature space.

    X_base : array-like
        The base samples. These are the samples from which new synthetic samples will be generated.

    X_base_weight : array-like
        The weights of the base samples. These weights represent the importance or reliability of the base samples.

    X_neighbor_weight : array-like
        The weights of the neighbor samples. These weights represent the importance or reliability of the neighbor samples.

    Returns:
    -------
    samples : ndarray
        The generated synthetic samples with weights applied.

    Notes:
    ------
    - The function assumes that the input arrays (X_neighbor, X_base, X_base_weight, X_neighbor_weight)
      are aligned, meaning that the i-th element in X_base corresponds to the i-th element in X_base_weight,
      and the i-th element in X_neighbor corresponds to the i-th element in X_neighbor_weight.
    - The function introduces randomness in the weight calculation to ensure diversity in the generated samples.
    """
    # Initialize lists to store interpolation weights and indices of samples to delete
    weights = []
    delete_index = []

    # Iterate over the number of base samples (each base sample will generate a new synthetic sample)
    for n in range(len(X_base)):
        # Check if both the base sample and its neighbor have non-zero weights
        if X_base_weight[n] != 0 and X_neighbor_weight[n] != 0:
            # If the base sample weight is greater than or equal to the neighbor sample weight
            if X_base_weight[n] >= X_neighbor_weight[n]:
                # Compute the interpolation proportion based on the neighbor weight and a random factor
                proportion = (X_neighbor_weight[n] / (X_base_weight[n] + X_neighbor_weight[n]) *
                             round(random.uniform(0, 1), len(str(len(X_base)))))
            # If the base sample weight is less than the neighbor sample weight
            elif X_base_weight[n] < X_neighbor_weight[n]:
                # Compute the interpolation proportion based on the neighbor weight and a random factor
                proportion = X_neighbor_weight[n] / (X_base_weight[n] + X_neighbor_weight[n])
                proportion = proportion + (1 - proportion) * (round(random.uniform(0, 1), len(str(len(X_base)))))
        
        # Append the computed proportion to the weights list
        weights.append(proportion)

    # Delete samples marked for deletion (if any)
    # Note: In the current implementation, `delete_index` is always empty, so no samples are deleted.
    X_neighbor = np.delete(X_neighbor, delete_index, axis=0)
    X_base = np.delete(X_base, delete_index, axis=0)

    # Reshape the weights array to match the dimensions of the base and neighbor samples
    weights = np.array(weights).reshape(int(len(weights)), 1)

    # Generate new synthetic samples by interpolating between the base and neighbor samples
    # using the computed weights
    samples = X_base + np.multiply(weights, X_neighbor - X_base)

    # Return the generated synthetic samples
    return samples