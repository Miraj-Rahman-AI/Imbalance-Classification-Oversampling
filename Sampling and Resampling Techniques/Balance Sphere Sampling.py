import warnings
warnings.filterwarnings("ignore")  # Ignore warnings to keep the output clean

from collections import Counter
from sklearn.cluster._kmeans import k_means
import numpy as np
import random
from sklearn.datasets import make_moons, make_circles
import pandas as pd
import matplotlib.pyplot as plt

# The following classes and functions are used to perform granular ball clustering and sampling.

class GBList:
    def __init__(self, data, alldata):
        """
        Initialize the GBList class.
        
        Parameters:
        - data: The dataset to be processed.
        - alldata: The entire dataset (used for reference).
        """
        self.data = data
        self.alldata = alldata
        self.granular_balls = [GranularBall(self.data)]  # Initialize with a single granular ball
        self.dict_granular_balls = {}  # Dictionary to store granular balls at different purity levels

    def init_granular_balls_dict(self, min_purity=0.51, max_purity=1.0, min_sample=1):
        """
        Initialize granular balls dictionary for a range of purity levels.
        
        Parameters:
        - min_purity: Minimum purity threshold.
        - max_purity: Maximum purity threshold.
        - min_sample: Minimum number of samples required for a granular ball.
        """
        for i in range(int((max_purity - min_purity) * 100) + 1):
            purity = i / 100 + min_purity
            self.init_granular_balls(purity, min_sample)
            self.dict_granular_balls[purity] = self.granular_balls.copy()

    def init_granular_balls(self, purity=1.0, min_sample=1):
        """
        Initialize granular balls based on the given purity threshold.
        
        Parameters:
        - purity: Purity threshold for granular balls.
        - min_sample: Minimum number of samples required for a granular ball.
        """
        ll = len(self.granular_balls)  # Current number of granular balls
        i = 0
        while True:
            if self.granular_balls[i].purity < purity and self.granular_balls[i].num > min_sample:
                split_clusters = self.granular_balls[i].split_clustering()
                if len(split_clusters) > 1:
                    self.granular_balls[i] = split_clusters[0]
                    self.granular_balls.extend(split_clusters[1:])
                    ll += len(split_clusters) - 1
                elif len(split_clusters) == 1:
                    i += 1
                else:
                    self.granular_balls.pop(i)
                    ll -= 1
            else:
                i += 1
            if i >= ll:
                for granular_ballsitem in self.granular_balls:
                    granular_ballsitem.get_radius()
                    granular_ballsitem.getBoundaryData()
                break


class GranularBall:
    def __init__(self, data):
        """
        Initialize the GranularBall class.
        
        Parameters:
        - data: The dataset to be processed.
        """
        self.data = data
        self.data_no_label = data[:, :-2]  # Data without labels
        self.num, self.dim = self.data_no_label.shape  # Number of samples and dimensions
        self.center = self.data_no_label.mean(0)  # Center of the granular ball
        self.label, self.purity = self.__get_label_and_purity()  # Label and purity of the granular ball
        self.init_center = self.random_center()  # Random initial center for clustering
        self.label_num = len(set(data[:, -2]))  # Number of unique labels
        self.boundaryData = None  # Boundary data points
        self.radius = None  # Radius of the granular ball

    def random_center(self):
        """
        Generate random centers for clustering.
        
        Returns:
        - center_array: Array of random centers.
        """
        center_array = np.empty(shape=[0, len(self.data_no_label[0, :])])
        for i in set(self.data[:, -2]):
            data_set = self.data_no_label[self.data[:, -2] == i, :]
            random_data = data_set[random.randrange(len(data_set)), :]
            center_array = np.append(center_array, [random_data], axis=0)
        return center_array

    def __get_label_and_purity(self):
        """
        Calculate the label and purity of the granular ball.
        
        Returns:
        - label: The dominant label in the granular ball.
        - purity: The purity of the granular ball.
        """
        count = Counter(self.data[:, -2])
        label = max(count, key=count.get)
        purity = count[label] / self.num
        return label, purity

    def get_radius(self):
        """
        Calculate the radius of the granular ball.
        """
        diffMat = np.tile(self.center, (self.num, 1)) - self.data_no_label
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        self.radius = distances.sum(axis=0) / self.num

    def split_clustering(self):
        """
        Split the granular ball into smaller clusters using K-means.
        
        Returns:
        - Clusterings: List of new granular balls.
        """
        Clusterings = []
        ClusterLists = k_means(X=self.data_no_label, init=self.init_center, n_clusters=self.label_num)
        data_label = ClusterLists[1]  # Get the cluster labels
        for i in range(self.label_num):
            Cluster_data = self.data[data_label == i, :]
            if len(Cluster_data) > 1:
                Cluster = GranularBall(Cluster_data)
                Clusterings.append(Cluster)
        return Clusterings

    def getBoundaryData(self):
        """
        Identify boundary data points within the granular ball.
        """
        if self.dim * 2 >= self.num:
            self.boundaryData = self.data
            return
        boundaryDataFalse = np.empty(shape=[0, self.dim])
        boundaryDataTrue = np.empty(shape=[0, self.dim + 2])
        for i in range(self.dim):
            centdataitem = np.tile(self.center, (1, 1))
            centdataitem[:, i] = centdataitem[:, i] + self.radius
            boundaryDataFalse = np.vstack((boundaryDataFalse, centdataitem))
            centdataitem = np.tile(self.center, (1, 1))
            centdataitem[:, i] = centdataitem[:, i] - self.radius
            boundaryDataFalse = np.vstack((boundaryDataFalse, centdataitem))
        list_path = []
        for boundaryDataItem in boundaryDataFalse:
            diffMat = np.tile(boundaryDataItem, (self.num, 1)) - self.data_no_label
            sqDiffMat = diffMat ** 2
            sqDistances = sqDiffMat.sum(axis=1)
            distances = sqDistances ** 0.5
            sortedDistances = distances.argsort()
            for i in range(self.num):
                if (self.data[sortedDistances[i]][-1] not in list_path and self.data[sortedDistances[i]][-2] == self.label):
                    boundaryDataTrue = np.vstack((boundaryDataTrue, self.data[sortedDistances[i]]))
                    list_path.append(self.data[sortedDistances[i]][-1])
                    break
        self.boundaryData = boundaryDataTrue


def main(train_data, train_label, purity=1.0):
    """
    Main function to perform granular ball clustering and sampling.
    
    Parameters:
    - train_data: Training data.
    - train_label: Training labels.
    - purity: Purity threshold for granular balls.
    
    Returns:
    - DataAll: Sampled data.
    - DataAllLabel: Labels of the sampled data.
    """
    numberSample, numberFeature = train_data.shape

    # Identify minority and majority classes
    number_set = set(train_label)
    label_1 = number_set.pop()
    label_2 = number_set.pop()

    if(train_label[(train_label == label_1)].shape[0] < train_label[(train_label == label_2)].shape[0]):
        less_label = label_1
        many_label = label_2
    else:
        less_label = label_2
        many_label = label_1

    DataAll = np.empty(shape=[0, numberFeature])
    DataAllLabel = []

    train = np.hstack((train_data, train_label.reshape(numberSample, 1)))
    index = np.array(range(0, numberSample)).reshape(numberSample, 1))
    train = np.hstack((train, index))

    granular_balls = GBList(train, train)
    granular_balls.init_granular_balls(purity=purity, min_sample=numberFeature * 2)
    init_l = granular_balls.granular_balls

    many_len = 0
    less_number = 0

    # Sample minority class
    for granular_ball in init_l:
        if granular_ball.label == less_label:
            data = granular_ball.boundaryData

            if granular_ball.purity >= purity:
                DataAll_index = []
                index_i = 0
                for data_item in granular_ball.data:
                    if data_item[numberFeature] == less_label:
                        DataAll_index.append(index_i)
                        less_number += 1
                    index_i += 1
                DataAll = np.vstack((DataAll, granular_ball.data[DataAll_index, : numberFeature]))
                DataAllLabel.extend(granular_ball.data[DataAll_index, numberFeature])
            else:
                DataAll = np.vstack((DataAll, data[:, : numberFeature]))
                DataAllLabel.extend(data[:, numberFeature])
                for data_item in data:
                    if data_item[numberFeature] == less_label:
                        less_number += 1
                    else:
                        many_len += 1

    dict = {}
    number = 0

    # Sample majority class
    for granular_ball in init_l:
        if (granular_ball.label == many_label):
            dict[number] = granular_ball.num
        number += 1
    sort_list = sorted(dict.items(), key=lambda item: item[1])
    gb_index = 0
    for sort_item in sort_list:
        granular_ball = init_l[sort_item[0]]
        if granular_ball.purity < purity:
            data = granular_ball.boundaryData
            DataAll = np.vstack((DataAll, data[:, : numberFeature]))
            DataAllLabel.extend(data[:, numberFeature])
            for data_item in data:
                if data_item[numberFeature] == less_label:
                    less_number += 1
                else:
                    many_len += 1
        else:
            if (granular_ball.dim * 2 * (len(dict) - gb_index) + many_len) < less_number:
                DataAll_index = []
                index_i = 0
                for data_item in granular_ball.data:
                    if data_item[numberFeature] == many_label:
                        DataAll_index.append(index_i)
                        many_len += 1
                    index_i += 1
                DataAll = np.vstack((DataAll, granular_ball.data[DataAll_index, : numberFeature]))
                DataAllLabel.extend(granular_ball.data[DataAll_index, numberFeature])
            else:
                data = granular_ball.boundaryData
                DataAll = np.vstack((DataAll, data[:, : numberFeature]))
                DataAllLabel.extend(data[:, numberFeature])
                many_len += data.shape[0]
        gb_index += 1

    return DataAll, DataAllLabel