# Import necessary libraries
from sklearn.preprocessing import MinMaxScaler  # For scaling data
from scipy.special import gamma  # Used to compute the volume of a ball
from math import log, math  # For logarithmic and mathematical operations
from sklearn.metrics import pairwise_distances  # For computing pairwise distances
from imblearn.over_sampling import KMeansSMOTE  # For SMOTE oversampling
import random  # For random operations
from sklearn.cluster import k_means  # For k-means clustering
import numpy as np  # For numerical operations
from matplotlib import pyplot as plt  # For plotting
from collections import Counter  # For counting occurrences
import warnings  # For handling warnings
warnings.filterwarnings("ignore")  # Ignore warnings

# Class to manage a list of Granular Balls
class GBList:
    def __init__(self, data) -> None:
        """
        Initialize the GBList with the given data.
        :param data: The dataset, where the last column is the label.
        """
        self.data: np.ndarray = data  # Store the dataset
        self.granular_balls: GranularBall = [GranularBall(self.data)]  # Initialize with one GranularBall

    def init_granular_balls(self, purity=1.0, min_sample=1, balls_list=None) -> None:
        """
        Initialize granular balls based on purity and minimum sample criteria.
        :param purity: Purity threshold for splitting balls.
        :param min_sample: Minimum number of samples required to split a ball.
        :param balls_list: List of granular balls to initialize.
        """
        if balls_list is None:
            balls_list = self.granular_balls

        length = len(balls_list)
        i = 0
        while True:
            # Check if the current ball meets the splitting criteria
            if balls_list[i].purity < purity and balls_list[i].num >= balls_list[i].label_num * min_sample:
                # Split the current granular ball
                split_clusters = balls_list[i].split_clustering()
                if len(split_clusters) > 1:  # If the ball can be split
                    balls_list[i] = split_clusters[0]
                    balls_list.extend(split_clusters[1:])
                    length += len(split_clusters) - 1
                elif len(split_clusters) == 1:  # If the ball cannot be split
                    i += 1  # Move to the next ball
                else:
                    balls_list.pop(i)  # Remove the ball if it's an anomaly
                    length -= 1
            else:
                i += 1  # Move to the next ball
            if i >= length:
                break

    def remove_overlap(self):
        """
        Remove overlapping granular balls.
        """
        print('Removing overlapping granular balls')
        length = len(self.granular_balls)
        pre, cur = 0, 1  # Indices for comparing balls

        while True:
            pre_ball = self.granular_balls[pre]
            cur_ball = self.granular_balls[cur]

            # Check if the balls overlap and have different labels
            if (pre_ball.label != cur_ball.label) and np.sum((pre_ball.center - cur_ball.center) ** 2, axis=0) ** 0.5 < (pre_ball.radius + cur_ball.radius):
                print("Overlap found:", pre, cur)

                # Split the larger ball
                if pre_ball.radius >= cur_ball.radius:
                    split_clusters = pre_ball.split_clustering()
                    len_split = len(split_clusters)
                    if len_split == 1:  # If the ball cannot be split
                        pre, cur = pre + 1, cur + 1
                    elif len_split > 1:
                        self.granular_balls[pre] = split_clusters[0]
                        self.granular_balls.extend(split_clusters[1:])
                        length += len_split - 1
                else:
                    split_clusters = cur_ball.split_clustering()
                    len_split = len(split_clusters)
                    if len_split == 1:  # If the ball cannot be split
                        pre, cur = pre + 1, cur + 1
                    elif len_split > 1:
                        self.granular_balls[cur] = split_clusters[0]
                        self.granular_balls.extend(split_clusters[1:])
                        length += len_split - 1
            else:
                pre, cur = pre + 1, cur + 1
            if cur >= length:
                break
        print('Overlap removal complete')

# Class representing a Granular Ball
class GranularBall:
    def __init__(self, data):
        """
        Initialize a Granular Ball with the given data.
        :param data: The dataset, where the last column is the label.
        """
        self.data: np.ndarray = data
        self.data_no_label: np.ndarray = data[:, :-1]
        self.num, self.dim = self.data_no_label.shape  # Number of samples and dimensions
        self.center: np.ndarray = self.data_no_label.mean(0)  # Center of the ball
        self.label_num: int = len(set(data[:, -1]))  # Number of unique labels
        self.label, self.purity, self.radius = self.info_of_ball()  # Label, purity, and radius

    def info_of_ball(self):
        """
        Compute the label, purity, and radius of the ball.
        :return: Label, purity, and radius.
        """
        count = Counter(self.data[:, -1])  # Count the labels
        label = max(count, key=count.get)  # Majority label
        purity = count[label] / self.num  # Purity of the ball
        radius = np.sum(np.sum((self.data_no_label - self.center) ** 2, axis=1) ** 0.5, axis=0) / self.num  # Radius of the ball
        return label, purity, radius

    def print_info(self) -> None:
        """
        Print information about the Granular Ball.
        """
        print('\n\n\t ************** Granular Ball Information **************')
        for k, v in self.__dict__.items():
            print(k, ':\t', v)
        print('\t ************** Granular Ball Information **************\n\n')

    def split_clustering(self):
        """
        Split the Granular Ball into smaller balls using clustering.
        :return: List of new Granular Balls.
        """
        center_array: np.ndarray = np.empty(shape=[0, len(self.data_no_label[0, :])])
        for i in set(self.data[:, -1]):  # For each label
            data_set = self.data_no_label[np.where(self.data[:, -1] == i)]
            random_data = data_set[random.randrange(len(data_set)), :]
            center_array = np.append(center_array, [random_data], axis=0)

        Clusterings = []
        ClusterLists = k_means(X=self.data_no_label, init=center_array, n_clusters=self.label_num,)
        data_label = ClusterLists[1]  # Get the cluster labels

        for i in set(data_label):
            Cluster_data = self.data[np.where(data_label == i)]
            if len(Cluster_data) > 1:
                Cluster = GranularBall(Cluster_data)
                Clusterings.append(Cluster)
        return Clusterings

# Class for Granular Ball-based Oversampling
class GB_OverSampling:
    def __init__(self, purity=0.5) -> None:
        """
        Initialize the oversampling class.
        :param purity: Purity threshold for granular balls.
        """
        self.sampling_strategy_: dict = {}
        self.purity: float = purity

    def _check_sampling_strategy_(self, y) -> dict:
        """
        Determine the sampling strategy for each class.
        :param y: Target labels.
        :return: Dictionary with the number of samples to generate for each class.
        """
        count = Counter(y)
        most_label = max(count, key=count.get)
        most_nums: int = count[most_label]
        ordered_dict: dict = {}
        for k, v in count.items():
            if k != most_label:
                ordered_dict[k] = most_nums - v
        return ordered_dict

    def _find_ball_sparsity(self, X: np.ndarray):
        """
        Compute the sparsity of a ball.
        :param X: Data points in the ball.
        :return: Sparsity value.
        """
        euclidean_distances = pairwise_distances(X, metric="euclidean",)
        for ind in range(X.shape[0]):
            euclidean_distances[ind, ind] = 0
        non_diag_elements = (X.shape[0] ** 2) - X.shape[0]
        mean_distance = euclidean_distances.sum() / non_diag_elements
        exponent = (math.log(X.shape[0], 1.6) ** 1.8 * 0.16)
        return (mean_distance ** exponent) / X.shape[0]

    def _find_ball_entropy(self, y: np.array):
        """
        Compute the entropy of a ball.
        :param y: Labels in the ball.
        :return: Entropy value.
        """
        num = len(y)
        label_counts = Counter(y)
        shannon_ent = 0
        for k, v in label_counts.items():
            prob = float(v)/num
            shannon_ent -= prob * log(prob, 2)
        return shannon_ent

    def _find_ball_density(self, radius, dims, min_num):
        """
        Compute the density of a ball.
        :param radius: Radius of the ball.
        :param dims: Number of dimensions.
        :param min_num: Number of minority samples.
        :return: Density value.
        """
        V = np.pi ** (dims/2.0) / gamma(dims/2.0 + 1) * radius**dims
        if min_num/V == np.inf:
            return 0
        return min_num / V

    def _find_ball_RD(self, ball):
        """
        Compute the relative density of samples in a ball.
        :param ball: The Granular Ball.
        :return: Relative density values.
        """
        RD = np.zeros(ball.data.shape[0])
        min = np.where(ball.data[:, -1] == ball.label)[0]
        maj = np.where(ball.data[:, -1] != ball.label)[0]
        data_min = ball.data_no_label[min]
        data_maj = ball.data_no_label[maj]

        for i in min:
            if len(maj) == 0:
                RD[i] = np.sum(np.sum((data_min - ball.data_no_label[i]) ** 2, axis=1) ** 0.5, axis=0) / (len(data_min)-1)
            elif len(min) == 1 and len(maj) != 0:
                continue
            else:
                homo = np.sum(np.sum((data_min - ball.data_no_label[i])**2, axis=1) ** 0.5, axis=0) / (len(data_min)-1)
                hete = np.sum(np.sum((data_maj - ball.data_no_label[i])**2, axis=1) ** 0.5, axis=0) / (len(data_maj))
                RD[i] = homo / hete
        return RD

    def _find_ball_WIE(self, RD):
        """
        Compute the weighted information entropy of a ball.
        :param RD: Relative density values.
        :return: Weighted information entropy.
        """
        if RD.sum() == 0:
            return 0
        RD_ratio = RD/RD.sum()
        WIE = 0
        for i in RD_ratio:
            if i != 0:
                WIE -= i * log(i, 2)
        return WIE/len(RD)

    def _fit_resample(self, X, y):
        """
        Perform the oversampling process.
        :param X: Features.
        :param y: Labels.
        :return: Resampled features and labels.
        """
        X_resampled = X.copy()
        y_resampled = y.copy()

        self.sampling_strategy_ = self._check_sampling_strategy_(y_resampled)

        nums, dims = X.shape
        data = np.hstack((X, y.reshape(nums, 1)))
        balls = GBList(data)
        balls.init_granular_balls(purity=self.purity, min_sample=dims+1)

        for class_sample, n_samples in self.sampling_strategy_.items():
            valid_balls = []
            balls_density = []
            balls_RD = []
            balls_WIE = []
            for ball in balls.granular_balls:
                if ball.label == class_sample:
                    RD = self._find_ball_RD(ball)
                    balls_WIE.append(self._find_ball_WIE(RD))
                    balls_RD.append(RD)
                    valid_balls.append(ball)

            balls_WIE = np.array(balls_WIE)
            balls_WIE[np.isnan(balls_WIE)] = 0
            Entropy_Threshold = np.mean(balls_WIE)

            for ind in range(len(valid_balls)-1, -1, -1):
                if balls_WIE[ind] < Entropy_Threshold:
                    del valid_balls[ind]
                    del balls_RD[ind]
                else:
                    balls_density.append(self._find_ball_density(valid_balls[ind].radius, dims, valid_balls[ind].num))

            weights_densi = np.array(balls_density)

            if not valid_balls:
                raise RuntimeError("Cannot find ball to SMOTE")

            X_new_res = np.zeros(dims)
            y_new_res = np.zeros(1)

            for index, ball in enumerate(valid_balls):
                ball_n_samples = int(math.ceil(n_samples * weights_densi[index]/weights_densi.sum()))
                balls_RD[index] = np.delete(balls_RD[index], np.where(balls_RD[index] == 0))
                index_array = balls_RD[index].argsort()
                distance = np.sort(balls_RD[index])

                for i in range(ball_n_samples):
                    seed_sample_ind = index_array[i % len(index_array)]
                    seed_sample = ball.data_no_label[seed_sample_ind]
                    seed_neigbor_ind = random.randint(0, len(index_array)-1)
                    while seed_neigbor_ind == index_array[i % len(index_array)]:
                        seed_neigbor_ind = random.randint(0, len(index_array)-1)
                    seed_neigbor = ball.data_no_label[seed_neigbor_ind]

                    new_center = (seed_sample * balls_RD[index][seed_sample_ind] + seed_neigbor * balls_RD[index][seed_neigbor_ind]) / (balls_RD[index][seed_sample_ind]+balls_RD[index][seed_neigbor_ind])
                    new_radius = np.sum((seed_sample - seed_neigbor)**2, axis=0) ** 0.5
                    X_new = np.random.normal(loc=new_center, scale=new_radius/dims, size=seed_sample.size)
                    X_new_res = np.vstack((X_new, X_new_res))

                y_new = np.full(ball_n_samples, class_sample)
                y_new_res = np.hstack((y_new, y_new_res))

            X_new_res = np.delete(X_new_res, -1, 0)
            y_new_res = np.delete(y_new_res, -1, 0)
            X_resampled = np.vstack((X_resampled, X_new_res))
            y_resampled = np.hstack((y_resampled, y_new_res))

        return X_resampled, y_resampled