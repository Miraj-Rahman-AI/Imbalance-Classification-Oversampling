import numpy as np  # For numerical computations and array manipulations

# Function to calculate the distance between two points using the p-norm
def distance(x, y, p_norm=1):
    """
    Calculate the distance between two points using the p-norm.
    
    Parameters:
    - x: First point (numpy array).
    - y: Second point (numpy array).
    - p_norm: The p-value for the norm (default is 1 for Manhattan distance).
    
    Returns:
    - The computed distance.
    """
    return np.sum(np.abs(x - y) ** p_norm) ** (1 / p_norm)


# Function to compute the Radial Basis Function (RBF) kernel
def rbf(d, gamma):
    """
    Compute the Radial Basis Function (RBF) kernel.
    
    Parameters:
    - d: Distance between two points.
    - gamma: Kernel parameter controlling the spread of the kernel.
    
    Returns:
    - The RBF kernel value.
    """
    if gamma == 0.0:
        return 0.0  # Avoid division by zero
    else:
        return np.exp(-(d / gamma) ** 2)  # RBF formula


# Function to compute the mutual class potential for a point
def mutual_class_potential(point, majority_points, minority_points, gamma):
    """
    Compute the mutual class potential for a point.
    
    Parameters:
    - point: The point for which the potential is computed.
    - majority_points: Array of majority class points.
    - minority_points: Array of minority class points.
    - gamma: Kernel parameter for the RBF function.
    
    Returns:
    - The computed mutual class potential.
    """
    result = 0.0

    # Add contributions from majority class points
    for majority_point in majority_points:
        result += rbf(distance(point, majority_point), gamma)

    # Subtract contributions from minority class points
    for minority_point in minority_points:
        result -= rbf(distance(point, minority_point), gamma)

    return result


# Function to generate possible translation directions
def generate_possible_directions(n_dimensions, excluded_direction=None):
    """
    Generate possible translation directions for a point.
    
    Parameters:
    - n_dimensions: Number of dimensions in the data.
    - excluded_direction: A direction to exclude (optional).
    
    Returns:
    - A list of possible directions as tuples (dimension, sign).
    """
    possible_directions = []

    # Generate directions for each dimension and sign
    for dimension in range(n_dimensions):
        for sign in [-1, 1]:
            if excluded_direction is None or (excluded_direction[0] != dimension or excluded_direction[1] != sign):
                possible_directions.append((dimension, sign))

    # Shuffle the directions to randomize the order
    np.random.shuffle(possible_directions)

    return possible_directions


# Class for Radial-Based Oversampling (RBO)
class RBO:
    """
    Radial-Based Oversampling (RBO) for handling imbalanced datasets.
    
    Parameters:
    - gamma: Kernel parameter for the RBF function.
    - step_size: Step size for translation.
    - n_steps: Number of steps for translation.
    - approximate_potential: Whether to approximate the potential using nearest neighbors.
    - n_nearest_neighbors: Number of nearest neighbors to consider for approximation.
    - minority_class: The minority class label (optional).
    - n: Number of synthetic samples to generate (optional).
    """
    def __init__(self, gamma=0.05, step_size=0.001, n_steps=500, approximate_potential=True,
                 n_nearest_neighbors=25, minority_class=None, n=None):
        self.gamma = gamma
        self.step_size = step_size
        self.n_steps = n_steps
        self.approximate_potential = approximate_potential
        self.n_nearest_neighbors = n_nearest_neighbors
        self.minority_class = minority_class
        self.n = n

    # Method to fit the oversampler and generate synthetic samples
    def fit_sample(self, X, y):
        """
        Fit the oversampler and generate synthetic samples for the minority class.
        
        Parameters:
        - X: Feature matrix.
        - y: Target labels.
        
        Returns:
        - A list of synthetic samples.
        """
        classes = np.unique(y)  # Get unique class labels

        # Determine the minority class if not provided
        if self.minority_class is None:
            sizes = [sum(y == c) for c in classes]
            minority_class = classes[np.argmin(sizes)]
        else:
            minority_class = self.minority_class

        # Separate minority and majority points
        minority_points = X[y == minority_class].copy()
        majority_points = X[y != minority_class].copy()

        # Determine the number of synthetic samples to generate
        if self.n is None:
            n = len(majority_points) - len(minority_points)
        else:
            n = self.n

        appended = []  # Store synthetic samples
        sorted_neighbors_indices = None
        considered_minority_points_indices = range(len(minority_points))

        # Distribute the number of synthetic samples across minority points
        n_synthetic_points_per_minority_object = {i: 0 for i in considered_minority_points_indices}
        for _ in range(n):
            idx = np.random.choice(considered_minority_points_indices)
            n_synthetic_points_per_minority_object[idx] += 1

        # Generate synthetic samples for each minority point
        for i in considered_minority_points_indices:
            if n_synthetic_points_per_minority_object[i] == 0:
                continue

            point = minority_points[i]

            # Approximate potential using nearest neighbors if enabled
            if self.approximate_potential:
                if sorted_neighbors_indices is None:
                    distance_vector = [distance(point, x) for x in X]
                    distance_vector[i] = -np.inf
                    indices = np.argsort(distance_vector)[:(self.n_nearest_neighbors + 1)]
                else:
                    indices = sorted_neighbors_indices[i][:(self.n_nearest_neighbors + 1)]

                closest_points = X[indices]
                closest_labels = y[indices]
                closest_minority_points = closest_points[closest_labels == minority_class]
                closest_majority_points = closest_points[closest_labels != minority_class]
            else:
                closest_minority_points = minority_points
                closest_majority_points = majority_points

            # Generate synthetic samples for the current minority point
            for _ in range(n_synthetic_points_per_minority_object[i]):
                translation = [0 for _ in range(len(point))]
                translation_history = [translation]
                potential = mutual_class_potential(point, closest_majority_points, closest_minority_points, self.gamma)
                possible_directions = generate_possible_directions(len(point))

                # Perform translation steps
                for _ in range(self.n_steps):
                    if len(possible_directions) == 0:
                        break

                    dimension, sign = possible_directions.pop()
                    modified_translation = translation.copy()
                    modified_translation[dimension] += sign * self.step_size
                    modified_potential = mutual_class_potential(point + modified_translation, closest_majority_points,
                                                                closest_minority_points, self.gamma)

                    # Update translation if potential improves
                    if np.abs(modified_potential) < np.abs(potential):
                        translation = modified_translation
                        translation_history.append(translation)
                        potential = modified_potential
                        possible_directions = generate_possible_directions(len(point), (dimension, -sign))

                appended.append(point + translation)

        return appended


# Class for Multi-Class Radial-Based Oversampling
class MultiClassRBO:
    """
    Multi-Class Radial-Based Oversampling for handling imbalanced datasets with multiple classes.
    
    Parameters:
    - gamma: Kernel parameter for the RBF function.
    - step_size: Step size for translation.
    - n_steps: Number of steps for translation.
    - approximate_potential: Whether to approximate the potential using nearest neighbors.
    - n_nearest_neighbors: Number of nearest neighbors to consider for approximation.
    - method: Oversampling method ('sampling' or 'complete').
    """
    def __init__(self, gamma=0.05, step_size=0.001, n_steps=500, approximate_potential=True,
                 n_nearest_neighbors=25, method='sampling'):
        assert method in ['sampling', 'complete']  # Validate the method

        self.gamma = gamma
        self.step_size = step_size
        self.n_steps = n_steps
        self.approximate_potential = approximate_potential
        self.n_nearest_neighbors = n_nearest_neighbors
        self.method = method

    # Method to fit the oversampler and generate synthetic samples for multi-class data
    def fit_sample(self, X, y):
        """
        Fit the oversampler and generate synthetic samples for multi-class data.
        
        Parameters:
        - X: Feature matrix.
        - y: Target labels.
        
        Returns:
        - Oversampled feature matrix and labels.
        """
        classes = np.unique(y)  # Get unique class labels
        sizes = np.array([float(sum(y == c)) for c in classes])
        indices = np.argsort(sizes)[::-1]  # Sort classes by size
        classes = classes[indices]
        observations = [X[y == c] for c in classes]  # Separate observations by class
        n_max = len(observations[0])  # Size of the majority class

        if self.method == 'sampling':
            # Oversample each minority class using sampling
            for i in range(1, len(classes)):
                cls = classes[i]
                n = n_max - len(observations[i])
                X_sample = [observations[i]]
                y_sample = [cls * np.ones(len(observations[i]))]

                for j in range(0, i):
                    indices = np.random.choice(range(len(observations[j])), int(n_max / i))
                    X_sample.append(observations[j][indices])
                    y_sample.append(classes[j] * np.ones(len(X_sample[-1])))

                oversampler = RBO(gamma=self.gamma, step_size=self.step_size, n_steps=self.n_steps,
                                  approximate_potential=self.approximate_potential,
                                  n_nearest_neighbors=self.n_nearest_neighbors, minority_class=cls, n=n)

                appended = oversampler.fit_sample(np.concatenate(X_sample), np.concatenate(y_sample))

                if len(appended) > 0:
                    observations[i] = np.concatenate([observations[i], appended])
        else:
            # Oversample each minority class using the complete dataset
            for i in range(1, len(classes)):
                cls = classes[i]
                n = n_max - len(observations[i])

                oversampler = RBO(gamma=self.gamma, step_size=self.step_size, n_steps=self.n_steps,
                                  approximate_potential=self.approximate_potential,
                                  n_nearest_neighbors=self.n_nearest_neighbors, minority_class=cls, n=n)

                appended = oversampler.fit_sample(X, y)

                if len(appended) > 0:
                    observations[i] = np.concatenate([observations[i], appended])

        # Combine oversampled observations and labels
        labels = [cls * np.ones(len(obs)) for obs, cls in zip(observations, classes)]

        return np.concatenate(observations), np.concatenate(labels)