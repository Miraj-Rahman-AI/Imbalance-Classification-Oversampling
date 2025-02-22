"""Class to perform over-sampling using Geometric SMOTE."""

# Import necessary libraries
import numpy as np  # For numerical operations
from numpy.linalg import norm  # For calculating vector norms
from sklearn.utils import check_random_state  # For managing random states
from imblearn.over_sampling.base import BaseOverSampler  # Base class for over-samplers
from imblearn.utils import check_neighbors_object, Substitution  # Utility functions
from imblearn.utils._docstring import _random_state_docstring  # For documentation

# Define valid selection strategies for Geometric SMOTE
SELECTION_STRATEGY = ('combined', 'majority', 'minority')

# Define a helper function to generate synthetic samples
def _make_geometric_sample(
    center, surface_point, truncation_factor, deformation_factor, random_state
):
    """
    Generate a synthetic sample inside the geometric region defined by the center and surface points.

    Parameters:
    ----------
    center : ndarray, shape (n_features, )
        The center point of the geometric region.

    surface_point : ndarray, shape (n_features, )
        A point on the surface of the geometric region.

    truncation_factor : float, optional (default=0.0)
        Controls the truncation of the geometric region. Values should be in the range [-1.0, 1.0].

    deformation_factor : float, optional (default=0.0)
        Controls the deformation of the geometric region. Values should be in the range [0.0, 1.0].

    random_state : int, RandomState instance, or None
        Controls the randomization of the algorithm.

    Returns:
    -------
    point : ndarray, shape (n_features, )
        A synthetically generated sample.
    """

    # If the center and surface point are the same, return the center
    if np.array_equal(center, surface_point):
        return center

    # Calculate the radius of the geometric region
    radius = norm(center - surface_point)

    # Generate a random point on the surface of a unit hyper-sphere
    normal_samples = random_state.normal(size=center.size)
    point_on_unit_sphere = normal_samples / norm(normal_samples)
    point = (random_state.uniform(size=1) ** (1 / center.size)) * point_on_unit_sphere

    # Calculate the parallel unit vector
    parallel_unit_vector = (surface_point - center) / norm(surface_point - center)

    # Apply truncation to the point
    close_to_opposite_boundary = (
        truncation_factor > 0
        and np.dot(point, parallel_unit_vector) < truncation_factor - 1
    )
    close_to_boundary = (
        truncation_factor < 0
        and np.dot(point, parallel_unit_vector) > truncation_factor + 1
    )
    if close_to_opposite_boundary or close_to_boundary:
        point -= 2 * np.dot(point, parallel_unit_vector) * parallel_unit_vector

    # Apply deformation to the point
    parallel_point_position = np.dot(point, parallel_unit_vector) * parallel_unit_vector
    perpendicular_point_position = point - parallel_point_position
    point = (
        parallel_point_position
        + (1 - deformation_factor) * perpendicular_point_position
    )

    # Translate the point to the correct position
    point = center + radius * point

    return point


# Define the GeometricSMOTE class
@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
)
class GeometricSMOTE(BaseOverSampler):
    """
    Class to perform over-sampling using Geometric SMOTE.

    Parameters:
    ----------
    {sampling_strategy}
    {random_state}

    truncation_factor : float, optional (default=0.0)
        Controls the truncation of the geometric region. Values should be in the range [-1.0, 1.0].

    deformation_factor : float, optional (default=0.0)
        Controls the deformation of the geometric region. Values should be in the range [0.0, 1.0].

    selection_strategy : str, optional (default='combined')
        The strategy for selecting points. Options are 'combined', 'majority', or 'minority'.

    k_neighbors : int or object, optional (default=5)
        The number of nearest neighbors to use when constructing synthetic samples.
        If an object, it must inherit from `sklearn.neighbors.base.KNeighborsMixin`.

    n_jobs : int, optional (default=1)
        The number of threads to use for parallel processing.

    Examples:
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from gsmote import GeometricSMOTE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> gsmote = GeometricSMOTE(random_state=1)
    >>> X_res, y_res = gsmote.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 900, 1: 900}})
    """

    def __init__(
        self,
        sampling_strategy='auto',
        random_state=None,
        truncation_factor=1.0,
        deformation_factor=0.0,
        selection_strategy='combined',
        k_neighbors=5,
        n_jobs=1,
    ):
        # Initialize the parent class
        super(GeometricSMOTE, self).__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.truncation_factor = truncation_factor
        self.deformation_factor = deformation_factor
        self.selection_strategy = selection_strategy
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Validate the estimator's parameters and create necessary attributes."""

        # Check and set the random state
        self.random_state_ = check_random_state(self.random_state)

        # Validate the selection strategy
        if self.selection_strategy not in SELECTION_STRATEGY:
            error_msg = (
                'Unknown selection_strategy for Geometric SMOTE algorithm. '
                'Choices are {}. Got {} instead.'
            )
            raise ValueError(
                error_msg.format(SELECTION_STRATEGY, self.selection_strategy)
            )

        # Create nearest neighbors object for the positive class (minority class)
        if self.selection_strategy in ('minority', 'combined'):
            self.nns_pos_ = check_neighbors_object(
                'nns_positive', self.k_neighbors, additional_neighbor=1
            )
            self.nns_pos_.set_params(n_jobs=self.n_jobs)

        # Create nearest neighbors object for the negative class (majority class)
        if self.selection_strategy in ('majority', 'combined'):
            self.nn_neg_ = check_neighbors_object('nn_negative', nn_object=1)
            self.nn_neg_.set_params(n_jobs=self.n_jobs)

    def _make_geometric_samples(self, X, y, pos_class_label, n_samples):
        """
        Generate synthetic samples inside the geometric region defined by nearest neighbors.

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            The feature matrix of the dataset.

        y : array-like, shape (n_samples, )
            The target labels of the dataset.

        pos_class_label : str or int
            The label of the minority class (positive class).

        n_samples : int
            The number of synthetic samples to generate.

        Returns:
        -------
        X_new : ndarray, shape (n_samples_new, n_features)
            The synthetically generated samples.

        y_new : ndarray, shape (n_samples_new, )
            The target labels for the synthetic samples.
        """

        # If no samples are to be generated, return empty arrays
        if n_samples == 0:
            return (
                np.array([], dtype=X.dtype).reshape(0, X.shape[1]),
                np.array([], dtype=y.dtype),
            )

        # Select the minority class samples
        X_pos = X[y == pos_class_label]

        # Force minority strategy if no negative class samples are present
        self.selection_strategy_ = (
            'minority' if len(X) == len(X_pos) else self.selection_strategy
        )

        # Minority or combined strategy
        if self.selection_strategy_ in ('minority', 'combined'):
            self.nns_pos_.fit(X_pos)
            points_pos = self.nns_pos_.kneighbors(X_pos)[1][:, 1:]
            samples_indices = self.random_state_.randint(
                low=0, high=len(points_pos.flatten()), size=n_samples)
            rows = np.floor_divide(samples_indices, points_pos.shape[1])
            cols = np.mod(samples_indices, points_pos.shape[1])

        # Majority or combined strategy
        if self.selection_strategy_ in ('majority', 'combined'):
            X_neg = X[y != pos_class_label]
            self.nn_neg_.fit(X_neg)
            points_neg = self.nn_neg_.kneighbors(X_pos)[1]
            if self.selection_strategy_ == 'majority':
                samples_indices = self.random_state_.randint(
                    low=0, high=len(points_neg.flatten()), size=n_samples)
                rows = np.floor_divide(samples_indices, points_neg.shape[1])
                cols = np.mod(samples_indices, points_neg.shape[1])

        # Generate new synthetic samples
        X_new = np.zeros((n_samples, X.shape[1]))  # Initialize array for new samples
        for ind, (row, col) in enumerate(zip(rows, cols)):

            # Define the center point
            center = X_pos[row]

            # Minority strategy
            if self.selection_strategy_ == 'minority':
                surface_point = X_pos[points_pos[row, col]]

            # Majority strategy
            elif self.selection_strategy_ == 'majority':
                surface_point = X_neg[points_neg[row, col]]

            # Combined strategy
            else:
                surface_point_pos = X_pos[points_pos[row, col]]
                surface_point_neg = X_neg[points_neg[row, 0]]
                radius_pos = norm(center - surface_point_pos)
                radius_neg = norm(center - surface_point_neg)
                surface_point = (
                    surface_point_neg if radius_pos > radius_neg else surface_point_pos)

            # Generate a new synthetic sample
            X_new[ind] = _make_geometric_sample(
                center,
                surface_point,
                self.truncation_factor,
                self.deformation_factor,
                self.random_state_,
            )

        # Create target labels for the new samples
        y_new = np.array([pos_class_label] * len(samples_indices))

        return X_new, y_new

    def _fit_resample(self, X, y):
        """
        Resample the dataset using Geometric SMOTE.

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            The feature matrix of the dataset.

        y : array-like, shape (n_samples, )
            The target labels of the dataset.

        Returns:
        -------
        X_resampled : ndarray, shape (n_samples_new, n_features)
            The resampled feature matrix.

        y_resampled : ndarray, shape (n_samples_new, )
            The resampled target labels.
        """

        # Validate the estimator's parameters
        self._validate_estimator()

        # Copy the original data
        X_resampled, y_resampled = X.copy(), y.copy()

        # Resample the dataset for each class
        for class_label, n_samples in self.sampling_strategy_.items():

            # Generate synthetic samples
            X_new, y_new = self._make_geometric_samples(X, y, class_label, n_samples)

            # Append the new samples to the resampled dataset
            X_resampled, y_resampled = (
                np.vstack((X_resampled, X_new)),
                np.hstack((y_resampled, y_new)),
            )

        return X_resampled, y_resampled