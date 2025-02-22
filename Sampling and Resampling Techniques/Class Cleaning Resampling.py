# Import necessary libraries
import numpy as np
from scipy.spatial import distance_matrix

# Define a function to calculate the distance between two points using the p-norm
def distance(x, y, p_norm=1):
    """
    Calculate the distance between two points x and y using the p-norm.

    Parameters:
    x (np.array): First point.
    y (np.array): Second point.
    p_norm (int): The p-norm to use for distance calculation (default is 1, which is Manhattan distance).

    Returns:
    float: The distance between x and y.
    """
    return np.sum(np.abs(x - y) ** p_norm) ** (1 / p_norm)

# Define a function to sample a point inside a sphere of a given radius in a high-dimensional space
def sample_inside_sphere(dimensionality, radius, p_norm=1):
    """
    Sample a point inside a sphere of a given radius in a high-dimensional space.

    Parameters:
    dimensionality (int): The number of dimensions of the space.
    radius (float): The radius of the sphere.
    p_norm (int): The p-norm to use for distance calculation (default is 1).

    Returns:
    np.array: A point inside the sphere.
    """
    # Generate a random direction vector
    direction_unit_vector = (2 * np.random.rand(dimensionality) - 1)
    # Normalize the direction vector to have a unit length according to the p-norm
    direction_unit_vector = direction_unit_vector / distance(direction_unit_vector, np.zeros(dimensionality), p_norm)
    # Scale the direction vector by a random radius within the sphere
    return direction_unit_vector * np.random.rand() * radius

# Define the CCR (Class Cleaning and Resampling) class
class CCR:
    def __init__(self, energy, cleaning_strategy='translate', selection_strategy='proportional', p_norm=1,
                 minority_class=None, n=None):
        """
        Initialize the CCR class.

        Parameters:
        energy (float): The energy parameter controlling the resampling process.
        cleaning_strategy (str): Strategy for cleaning the majority class ('ignore', 'translate', 'remove').
        selection_strategy (str): Strategy for selecting minority class samples ('proportional', 'random').
        p_norm (int): The p-norm to use for distance calculation (default is 1).
        minority_class (int): The minority class label (default is None, which means it will be determined automatically).
        n (int): The number of samples to generate (default is None, which means it will be determined automatically).
        """
        assert cleaning_strategy in ['ignore', 'translate', 'remove']
        assert selection_strategy in ['proportional', 'random']

        self.energy = energy
        self.cleaning_strategy = cleaning_strategy
        self.selection_strategy = selection_strategy
        self.p_norm = p_norm
        self.minority_class = minority_class
        self.n = n

    def fit_sample(self, X, y):
        """
        Fit the CCR model and resample the data.

        Parameters:
        X (np.array): The feature matrix.
        y (np.array): The label vector.

        Returns:
        np.array: The resampled feature matrix.
        np.array: The resampled label vector.
        """
        # Determine the minority class if not specified
        if self.minority_class is None:
            classes = np.unique(y)
            sizes = [sum(y == c) for c in classes]
            minority_class = classes[np.argmin(sizes)]
        else:
            minority_class = self.minority_class

        # Separate the minority and majority class points
        minority_points = X[y == minority_class].copy()
        majority_points = X[y != minority_class].copy()
        minority_labels = y[y == minority_class].copy()
        majority_labels = y[y != minority_class].copy()

        # Determine the number of samples to generate if not specified
        if self.n is None:
            n = len(majority_points) - len(minority_points)
        else:
            n = self.n

        # Calculate the distance matrix between minority and majority points
        distances = distance_matrix(minority_points, majority_points, self.p_norm)

        # Initialize arrays to store radii, translations, and indices of kept points
        radii = np.zeros(len(minority_points))
        translations = np.zeros(majority_points.shape)
        kept_indices = np.full(len(majority_points), True)

        # Iterate over each minority point to calculate the radius and translations
        for i in range(len(minority_points)):
            minority_point = minority_points[i]
            remaining_energy = self.energy
            radius = 0.0
            sorted_distances = np.argsort(distances[i])
            n_majority_points_within_radius = 0

            # Calculate the radius for the current minority point
            while True:
                if n_majority_points_within_radius == len(majority_points):
                    if n_majority_points_within_radius == 0:
                        radius_change = remaining_energy / (n_majority_points_within_radius + 1)
                    else:
                        radius_change = remaining_energy / n_majority_points_within_radius

                    radius += radius_change
                    break

                radius_change = remaining_energy / (n_majority_points_within_radius + 1)

                if distances[i, sorted_distances[n_majority_points_within_radius]] >= radius + radius_change:
                    radius += radius_change
                    break
                else:
                    if n_majority_points_within_radius == 0:
                        last_distance = 0.0
                    else:
                        last_distance = distances[i, sorted_distances[n_majority_points_within_radius - 1]]

                    radius_change = distances[i, sorted_distances[n_majority_points_within_radius]] - last_distance
                    radius += radius_change
                    remaining_energy -= radius_change * (n_majority_points_within_radius + 1)
                    n_majority_points_within_radius += 1

            radii[i] = radius

            # Apply translations to the majority points within the calculated radius
            for j in range(n_majority_points_within_radius):
                majority_point = majority_points[sorted_distances[j]]
                d = distances[i, sorted_distances[j]]

                while d < 1e-20:
                    majority_point += (1e-6 * np.random.rand(len(majority_point)) + 1e-6) * \
                                      np.random.choice([-1.0, 1.0], len(majority_point))
                    d = distance(minority_point, majority_point)

                translation = (radius - d) / d * (majority_point - minority_point)
                translations[sorted_distances[j]] += translation
                kept_indices[sorted_distances[j]] = False

        # Apply the cleaning strategy to the majority points
        if self.cleaning_strategy == 'translate':
            majority_points += translations
        elif self.cleaning_strategy == 'remove':
            majority_points = majority_points[kept_indices]
            majority_labels = majority_labels[kept_indices]

        # Generate synthetic samples for the minority class
        appended = []

        if self.selection_strategy == 'proportional':
            for i in range(len(minority_points)):
                minority_point = minority_points[i]
                n_synthetic_samples = int(np.round(1.0 / (radii[i] * np.sum(1.0 / radii)) * n)
                r = radii[i]

                for _ in range(n_synthetic_samples):
                    appended.append(minority_point + sample_inside_sphere(len(minority_point), r, self.p_norm))
        elif self.selection_strategy == 'random':
            for i in np.random.choice(range(len(minority_points)), n):
                minority_point = minority_points[i]
                r = radii[i]

                appended.append(minority_point + sample_inside_sphere(len(minority_point), r, self.p_norm))

        # Combine the resampled points and labels
        if len(appended) > 0:
            points = np.concatenate([majority_points, minority_points, appended])
            labels = np.concatenate([majority_labels, minority_labels, np.tile([minority_class], len(appended))])
        else:
            points = np.concatenate([majority_points, minority_points])
            labels = np.concatenate([majority_labels, minority_labels])

        return points, labels

# Define the MultiClassCCR class for handling multi-class datasets
class MultiClassCCR:
    def __init__(self, energy, cleaning_strategy='translate', selection_strategy='proportional', p_norm=1,
                 method='sampling'):
        """
        Initialize the MultiClassCCR class.

        Parameters:
        energy (float): The energy parameter controlling the resampling process.
        cleaning_strategy (str): Strategy for cleaning the majority class ('ignore', 'translate', 'remove').
        selection_strategy (str): Strategy for selecting minority class samples ('proportional', 'random').
        p_norm (int): The p-norm to use for distance calculation (default is 1).
        method (str): The method for handling multi-class datasets ('sampling', 'complete').
        """
        assert cleaning_strategy in ['ignore', 'translate', 'remove']
        assert selection_strategy in ['proportional', 'random']
        assert method in ['sampling', 'complete']

        self.energy = energy
        self.cleaning_strategy = cleaning_strategy
        self.selection_strategy = selection_strategy
        self.p_norm = p_norm
        self.method = method

    def fit_sample(self, X, y):
        """
        Fit the MultiClassCCR model and resample the data.

        Parameters:
        X (np.array): The feature matrix.
        y (np.array): The label vector.

        Returns:
        np.array: The resampled feature matrix.
        np.array: The resampled label vector.
        """
        # Determine the unique classes and their sizes
        classes = np.unique(y)
        sizes = np.array([sum(y == c) for c in classes])
        indices = np.argsort(sizes)[::-1]
        classes = classes[indices]
        observations = {c: X[y == c] for c in classes}
        n_max = max(sizes)

        # Handle multi-class resampling based on the chosen method
        if self.method == 'sampling':
            for i in range(1, len(classes)):
                current_class = classes[i]
                n = n_max - len(observations[current_class])

                used_observations = {}
                unused_observations = {}

                for j in range(0, i):
                    all_indices = list(range(len(observations[classes[j]])))
                    used_indices = np.random.choice(all_indices, int(n_max / i), replace=False)

                    used_observations[classes[j]] = [
                        observations[classes[j]][idx] for idx in all_indices if idx in used_indices
                    ]
                    unused_observations[classes[j]] = [
                        observations[classes[j]][idx] for idx in all_indices if idx not in used_indices
                    ]

                used_observations[current_class] = observations[current_class]
                unused_observations[current_class] = []

                for j in range(i + 1, len(classes)):
                    used_observations[classes[j]] = []
                    unused_observations[classes[j]] = observations[classes[j]]

                unpacked_points, unpacked_labels = MultiClassCCR._unpack_observations(used_observations)

                ccr = CCR(energy=self.energy, cleaning_strategy=self.cleaning_strategy,
                          selection_strategy=self.selection_strategy, p_norm=self.p_norm,
                          minority_class=current_class, n=n)

                oversampled_points, oversampled_labels = ccr.fit_sample(unpacked_points, unpacked_labels)

                observations = {}

                for cls in classes:
                    class_oversampled_points = oversampled_points[oversampled_labels == cls]
                    class_unused_points = unused_observations[cls]

                    if len(class_oversampled_points) == 0 and len(class_unused_points) == 0:
                        observations[cls] = np.array([])
                    elif len(class_oversampled_points) == 0:
                        observations[cls] = class_unused_points
                    elif len(class_unused_points) == 0:
                        observations[cls] = class_oversampled_points
                    else:
                        observations[cls] = np.concatenate([class_oversampled_points, class_unused_points])
        else:
            for i in range(1, len(classes)):
                current_class = classes[i]
                n = n_max - len(observations[current_class])

                unpacked_points, unpacked_labels = MultiClassCCR._unpack_observations(observations)

                ccr = CCR(energy=self.energy, cleaning_strategy=self.cleaning_strategy,
                          selection_strategy=self.selection_strategy, p_norm=self.p_norm,
                          minority_class=current_class, n=n)

                oversampled_points, oversampled_labels = ccr.fit_sample(unpacked_points, unpacked_labels)

                observations = {cls: oversampled_points[oversampled_labels == cls] for cls in classes}

        unpacked_points, unpacked_labels = MultiClassCCR._unpack_observations(observations)

        return unpacked_points, unpacked_labels

    @staticmethod
    def _unpack_observations(observations):
        """
        Unpack the observations dictionary into points and labels.

        Parameters:
        observations (dict): A dictionary where keys are class labels and values are arrays of points.

        Returns:
        np.array: The concatenated feature matrix.
        np.array: The concatenated label vector.
        """
        unpacked_points = []
        unpacked_labels = []

        for cls in observations.keys():
            if len(observations[cls]) > 0:
                unpacked_points.append(observations[cls])
                unpacked_labels.append(np.tile([cls], len(observations[cls])))

        unpacked_points = np.concatenate(unpacked_points)
        unpacked_labels = np.concatenate(unpacked_labels)

        return unpacked_points, unpacked_labels