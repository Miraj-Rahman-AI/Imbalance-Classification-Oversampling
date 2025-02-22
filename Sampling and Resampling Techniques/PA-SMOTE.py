# Import necessary libraries
import numpy as np  # For numerical operations
import torch  # For tensor operations and automatic differentiation
from sklearn.cluster import KMeans  # For clustering data points
from tqdm import tqdm  # For progress bars during iterations

# Define a Radial Basis Function (RBF) kernel
def rbf(d, gamma):
    """
    Compute the RBF kernel value for a given distance and gamma.
    
    Parameters:
    - d: Distance between two points.
    - gamma: Parameter controlling the width of the RBF kernel.
    
    Returns:
    - RBF kernel value.
    """
    return torch.exp(-d.div(gamma).pow(2))

# Compute the potential of a point with respect to a set of points using the RBF kernel
def potential(x, points, gamma):
    """
    Compute the potential of a point x with respect to a set of points using the RBF kernel.
    
    Parameters:
    - x: The point for which potential is computed.
    - points: The set of points influencing the potential.
    - gamma: Parameter controlling the width of the RBF kernel.
    
    Returns:
    - The computed potential value.
    """
    result = 0.0

    for point in points:
        result += rbf(torch.dist(x, point), gamma)

    return result

# Normalize a vector to have unit length
def normalize(v):
    """
    Normalize a vector to have unit length (L2 norm).
    
    Parameters:
    - v: The vector to be normalized.
    
    Returns:
    - The normalized vector.
    """
    return v.div(torch.norm(v, p=2).detach())

# Compute the normalized potential of anchors with respect to a set of points
def normalized_potential(anchors, points, gamma):
    """
    Compute the normalized potential of anchors with respect to a set of points.
    
    Parameters:
    - anchors: The anchor points.
    - points: The set of points influencing the potential.
    - gamma: Parameter controlling the width of the RBF kernel.
    
    Returns:
    - The normalized potential values.
    """
    result = torch.zeros(anchors.shape[0])

    for i, anchor in enumerate(anchors):
        result[i] = potential(anchor, points, gamma)

    result = normalize(result)

    return result

# Compute the regularization term to penalize deviations from starting positions
def regularization_term(prototypes, starting_positions, gamma):
    """
    Compute the regularization term to penalize deviations from starting positions.
    
    Parameters:
    - prototypes: The current positions of prototypes.
    - starting_positions: The initial positions of prototypes.
    - gamma: Parameter controlling the width of the RBF kernel.
    
    Returns:
    - The regularization term value.
    """
    result = 0.0

    for prototype, starting_position in zip(prototypes, starting_positions):
        result += rbf(torch.dist(prototype, starting_position), gamma)

    return result

# Define the loss function for the optimization problem
def loss_function(anchors, prototypes, starting_positions, reference_potential, gamma, lambd=0.0):
    """
    Compute the loss function for the optimization problem.
    
    Parameters:
    - anchors: The anchor points.
    - prototypes: The current positions of prototypes.
    - starting_positions: The initial positions of prototypes.
    - reference_potential: The reference potential values.
    - gamma: Parameter controlling the width of the RBF kernel.
    - lambd: Regularization parameter.
    
    Returns:
    - The computed loss value.
    """
    loss = ((reference_potential - normalized_potential(anchors, prototypes, gamma)) ** 2).mean()

    if lambd > 0.0:
        loss += lambd * regularization_term(prototypes, starting_positions, gamma)

    return loss

# Define an abstract class for Potential Anchors (PA) sampling
class AbstractPA:
    def __init__(self, kind, gamma=0.5, lambd=0.0, n_anchors=10, learning_rate=0.001,
                 iterations=200, epsilon=1e-4, minority_class=None, n=None, ratio=None,
                 random_state=None, device=torch.device('cpu')):
        """
        Initialize the AbstractPA class.
        
        Parameters:
        - kind: Type of sampling ('oversample' or 'undersample').
        - gamma: Parameter controlling the width of the RBF kernel.
        - lambd: Regularization parameter.
        - n_anchors: Number of anchor points.
        - learning_rate: Learning rate for optimization.
        - iterations: Number of optimization iterations.
        - epsilon: Small noise added to initial prototypes.
        - minority_class: The minority class label.
        - n: Number of samples to generate.
        - ratio: Ratio of samples to generate.
        - random_state: Seed for random number generation.
        - device: Device to run computations on (CPU or GPU).
        """
        assert kind in ['oversample', 'undersample']

        self.kind = kind
        self.gamma = gamma
        self.lambd = lambd
        self.n_anchors = n_anchors
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.epsilon = epsilon
        self.minority_class = minority_class
        self.n = n
        self.ratio = ratio
        self.random_state = random_state
        self.device = device

        self._anchors = None
        self._prototypes = None
        self._loss = None

    def sample(self, X, y):
        """
        Perform sampling (oversampling or undersampling) on the dataset.
        
        Parameters:
        - X: Feature matrix.
        - y: Label vector.
        
        Returns:
        - X_: Sampled feature matrix.
        - y_: Sampled label vector.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.random.manual_seed(self.random_state)

        classes = np.unique(y)

        assert len(classes) == 2

        if self.minority_class is None:
            sizes = [sum(y == c) for c in classes]

            minority_class = classes[np.argmin(sizes)]
            majority_class = classes[np.argmax(sizes)]
        else:
            minority_class = self.minority_class

            if classes[0] != minority_class:
                majority_class = classes[0]
            else:
                majority_class = classes[1]

        minority_points = X[y == minority_class]
        majority_points = X[y == majority_class]

        assert not ((self.n is not None) and (self.ratio is not None))

        if self.n is not None:
            n = self.n
        elif self.ratio is not None:
            if self.kind == 'oversample':
                n = int(self.ratio * (len(majority_points) - len(minority_points)))
            else:
                n = len(majority_points) - int(self.ratio * (len(majority_points) - len(minority_points)))
        else:
            if self.kind == 'oversample':
                n = len(majority_points) - len(minority_points)
            else:
                n = len(majority_points)

        if (self.kind == 'oversample' and n == 0) or (self.kind != 'oversample' and n == len(majority_points)):
            return X, y

        self._anchors = torch.tensor(
            KMeans(n_clusters=self.n_anchors, random_state=self.random_state).fit(X).cluster_centers_,
            device=self.device, requires_grad=False, dtype=torch.float
        )

        if self.kind == 'oversample':
            reference_points = minority_points
        else:
            reference_points = majority_points

        indices = np.random.randint(reference_points.shape[0], size=n)

        self._prototypes = torch.tensor(
            reference_points[indices, :] + np.random.normal(scale=self.epsilon, size=(n, reference_points.shape[1])),
            device=self.device, requires_grad=True, dtype=torch.float
        )

        starting_positions = torch.tensor(
            reference_points[indices, :], device=self.device, requires_grad=False, dtype=torch.float
        )

        reference_points = torch.tensor(reference_points, device=self.device, requires_grad=False, dtype=torch.float)
        reference_potential = normalized_potential(self._anchors, reference_points, self.gamma)

        optimizer = torch.optim.Adam([self._prototypes], lr=self.learning_rate)

        self._loss = []

        with tqdm(total=self.iterations) as pbar:
            for i in range(self.iterations):
                loss = loss_function(
                    self._anchors, self._prototypes, starting_positions,
                    reference_potential, self.gamma, self.lambd
                )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                self._loss.append(loss.data.item())

                if np.isnan(self._loss[-1]):
                    raise RuntimeError('Convergence error (loss is NaN).')

                pbar.set_description(f'Iteration {i + 1}: loss = {self._loss[-1]:.7f}')
                pbar.update()

        self._anchors = self._anchors.cpu().detach().numpy()
        self._prototypes = self._prototypes.cpu().detach().numpy()

        if self.kind == 'oversample':
            X_ = np.concatenate([X, self._prototypes])
            y_ = np.concatenate([y, minority_class * np.ones(n)])
        else:
            X_ = np.concatenate([minority_points, self._prototypes])
            y_ = np.concatenate([minority_class * np.ones(len(minority_points)), majority_class * np.ones(n)])

        return X_, y_

# Define a class for Potential Anchors Oversampling (PAO)
class PAO(AbstractPA):
    def __init__(self, gamma=0.5, lambd=10.0, n_anchors=10, learning_rate=0.001,
                 iterations=200, epsilon=1e-4, minority_class=None, n=None, ratio=None,
                 random_state=None, device=torch.device('cpu')):
        """
        Initialize the PAO class for oversampling.
        
        Parameters:
        - gamma: Parameter controlling the width of the RBF kernel.
        - lambd: Regularization parameter.
        - n_anchors: Number of anchor points.
        - learning_rate: Learning rate for optimization.
        - iterations: Number of optimization iterations.
        - epsilon: Small noise added to initial prototypes.
        - minority_class: The minority class label.
        - n: Number of samples to generate.
        - ratio: Ratio of samples to generate.
        - random_state: Seed for random number generation.
        - device: Device to run computations on (CPU or GPU).
        """
        super().__init__(
            kind='oversample', gamma=gamma, lambd=lambd, n_anchors=n_anchors,
            learning_rate=learning_rate, iterations=iterations,
            epsilon=epsilon, minority_class=minority_class,
            n=n, ratio=ratio, random_state=random_state, device=device
        )

# Define a class for Potential Anchors Undersampling (PAU)
class PAU(AbstractPA):
    def __init__(self, gamma=0.5, lambd=0.0, n_anchors=10, learning_rate=0.001,
                 iterations=200, epsilon=1e-4, minority_class=None, n=None, ratio=None,
                 random_state=None, device=torch.device('cpu')):
        """
        Initialize the PAU class for undersampling.
        
        Parameters:
        - gamma: Parameter controlling the width of the RBF kernel.
        - lambd: Regularization parameter.
        - n_anchors: Number of anchor points.
        - learning_rate: Learning rate for optimization.
        - iterations: Number of optimization iterations.
        - epsilon: Small noise added to initial prototypes.
        - minority_class: The minority class label.
        - n: Number of samples to generate.
        - ratio: Ratio of samples to generate.
        - random_state: Seed for random number generation.
        - device: Device to run computations on (CPU or GPU).
        """
        super().__init__(
            kind='undersample', gamma=gamma, lambd=lambd, n_anchors=n_anchors,
            learning_rate=learning_rate, iterations=iterations,
            epsilon=epsilon, minority_class=minority_class,
            n=n, ratio=ratio, random_state=random_state, device=device
        )

# Define a class for Potential Anchors (PA) combining both oversampling and undersampling
class PA:
    def __init__(self, ratio=0.1, gamma=0.5, lambda_pao=10.0, lambda_pau=0.0, n_anchors=10,
                 learning_rate=0.001, iterations=200, epsilon=1e-4, minority_class=None,
                 random_state=None, device=torch.device('cpu')):
        """
        Initialize the PA class combining both oversampling and undersampling.
        
        Parameters:
        - ratio: Ratio of samples to generate.
        - gamma: Parameter controlling the width of the RBF kernel.
        - lambda_pao: Regularization parameter for oversampling.
        - lambda_pau: Regularization parameter for undersampling.
        - n_anchors: Number of anchor points.
        - learning_rate: Learning rate for optimization.
        - iterations: Number of optimization iterations.
        - epsilon: Small noise added to initial prototypes.
        - minority_class: The minority class label.
        - random_state: Seed for random number generation.
        - device: Device to run computations on (CPU or GPU).
        """
        assert 0 <= ratio <= 1

        self.ratio = ratio
        self.gamma = gamma
        self.lambda_pao = lambda_pao
        self.lambda_pau = lambda_pau
        self.n_anchors = n_anchors
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.epsilon = epsilon
        self.minority_class = minority_class
        self.random_state = random_state
        self.device = device

        self.pao = PAO(
            gamma=gamma, lambd=lambda_pao, n_anchors=n_anchors, learning_rate=learning_rate,
            iterations=iterations, epsilon=epsilon, minority_class=minority_class,
            random_state=random_state, device=device
        )

        self.pau = PAU(
            gamma=gamma, lambd=lambda_pau, n_anchors=n_anchors, learning_rate=learning_rate,
            iterations=iterations, epsilon=epsilon, minority_class=minority_class,
            random_state=random_state, device=device
        )

    def sample(self, X, y):
        """
        Perform sampling (oversampling and undersampling) on the dataset.
        
        Parameters:
        - X: Feature matrix.
        - y: Label vector.
        
        Returns:
        - X_: Sampled feature matrix.
        - y_: Sampled label vector.
        """
        classes = np.unique(y)

        assert len(classes) == 2

        if self.minority_class is None:
            sizes = [sum(y == c) for c in classes]

            minority_class = classes[np.argmin(sizes)]
            majority_class = classes[np.argmax(sizes)]
        else:
            minority_class = self.minority_class

            if classes[0] != minority_class:
                majority_class = classes[0]
            else:
                majority_class = classes[1]

        minority_points = X[y == minority_class]
        majority_points = X[y == majority_class]

        n = len(majority_points) - len(minority_points)

        self.pao.n = int(self.ratio * n)
        self.pau.n = len(majority_points) - int((1 - self.ratio) * n)

        self.pao.sample(X, y)
        self.pau.sample(X, y)

        if self.pau._prototypes is None:
            X_, y_ = X, y
        else:
            X_ = np.concatenate([minority_points, self.pau._prototypes])
            y_ = np.concatenate([
                minority_class * np.ones(len(minority_points)),
                majority_class * np.ones(len(self.pau._prototypes))
            ])

        if self.pao._prototypes is not None:
            X_ = np.concatenate([X_, self.pao._prototypes])
            y_ = np.concatenate([y_, minority_class * np.ones(len(self.pao._prototypes))])

        return X_, y_