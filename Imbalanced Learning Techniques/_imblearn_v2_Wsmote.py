
import math
from collections import Counter
import numpy as np
from scipy import sparse
from scipy.spatial import distance
from sklearn.base import clone
from sklearn.cluster import MiniBatchKMeans,KMeans,DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.utils import check_random_state
from sklearn.utils import _safe_indexing
from sklearn.utils import check_array
from sklearn.utils import check_X_y
# from sklearn.utils.sparsefuncs_fast import csr_mean_variance_axis0
# from sklearn.utils.sparsefuncs_fast import csc_mean_variance_axis0
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.exceptions import raise_isinstance_error
from imblearn.utils import check_neighbors_object
from imblearn.utils import check_target_type
from imblearn.utils import Substitution
from imblearn.utils._docstring import _n_jobs_docstring
from imblearn.utils._docstring import _random_state_docstring
import random


#-----------------------------------------------------------------------------------
def make_samples_zhou(
        X, y_dtype, y_type, nn_data, nn_num, n_samples,
        new_n_maj, danger_and_safe=None, step_size=1.0, kind=None, mother_point=None
):

    """
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Points from which the points will be created.

        y_dtype : dtype
            The data type of the targets.

        y_type : str or int
            The minority target value, just so the function can return the
            target values for the synthetic variables with correct length in
            a clear format.

        nn_data : ndarray of shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used

        nn_num : ndarray of shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

        n_samples : int
            The number of samples to generate.

        step_size : float, default=1.0
            The step size to create samples.

        Returns
        -------
        X_new : {ndarray, sparse matrix} of shape (n_samples_new, n_features)
            Synthetically generated samples.

        y_new : ndarray of shape (n_samples_new,)
            Target values for synthetic samples.
    """

    random_state = check_random_state(42)
    samples_indices = random_state.randint(
        low=0, high=nn_num.size, size=n_samples
    )

    # np.newaxis for backwards compatability with random_state
    steps = step_size * random_state.uniform(size=n_samples)[:, np.newaxis]
    rows = np.floor_divide(samples_indices, nn_num.shape[1])  
    cols = np.mod(samples_indices, nn_num.shape[1])

    if kind == None:
        X_new = generate_samples_zhou(X, nn_data, nn_num, rows, cols, steps, n_samples,
                                    new_n_maj, danger_and_safe)
        y_new = np.full(len(X_new), fill_value=y_type, dtype=y_dtype)

        return X_new, y_new
    elif kind == 'svm':
        X_new = generate_samples_zhou_svm_smote(X, nn_data, nn_num, rows, cols, steps, n_samples,
                                                new_n_maj, danger_and_safe, mother_point)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new
    elif kind == 'borderline':
        X_new = generate_samples_zhou_borderline(X, nn_data, nn_num, rows, cols, steps, n_samples,
                                                new_n_maj, danger_and_safe)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new


def generate_samples_zhou(X, nn_data:"Minority Class Matrix", nn_num, rows, cols, steps, n_samples:"Number to insert",
                        new_n_maj, danger_and_safe):
'''
zhou
# n = (number to be inserted) // number of minority classes, take integer, number to be inserted for each point
# m = (number to be inserted)% number of minority classes, take remainder, extra for random
#Traverse the boundary and safe points, find the corresponding KNN points according to the neighbor point index matrix, randomly extract with replacement N times, subtract the coordinates of two points, calculate the distance according to the weight ratio to interpolate,
#The remaining m points are used for random extraction without replacement, calculate the weight ratio of the extracted points and the neighbor points, and each point is only inserted once
'''

    n = n_samples // len(nn_data)
    m = n_samples % len(nn_data)
    weidu = X.shape[1]              
    X_new_1 = np.zeros(weidu)      

    for nn in range(len(nn_data)):  

        num = nn_num[nn].tolist() 
        if not isinstance(nn_data, list):nn_data = nn_data.tolist()

        nn_point = nn_data[nn]  
        nn_weight = new_n_maj[nn]  
        delet_index = 0
        for i in range(n):  
            
            # random_point = random.choice(num)  
            random_point = num[i%len(num)]          

            # random_point_index = num.index(random_point)        
            random_point_weight = new_n_maj[random_point]  
            # print('random_point:\t',random_point)
            random_point_data = nn_data[random_point]  

            if nn_weight != 0 and random_point_weight != 0: 

                if nn_weight + random_point_weight == 0:
                    print(nn_weight,random_point_weight)

                proportion = (random_point_weight / (nn_weight + random_point_weight))  
                if nn_weight >= random_point_weight:
                    X_new_zhou = np.array(nn_point) + (
                                np.array(random_point_data) - np.array(nn_point)) * proportion * round(
                        random.uniform(0, 1), len(str(n_samples)))
                elif nn_weight < random_point_weight:
                    X_new_zhou = np.array(random_point_data) + (np.array(nn_point) - np.array(random_point_data)) * (
                                1 - proportion) * round(random.uniform(0, 1), len(str(n_samples)))

                X_new_1 = np.vstack((X_new_zhou, X_new_1))
            elif (nn_weight + random_point_weight) == 0:  
                # proportion = 0
                # X_new_zhou = np.array(nn_point)
                delet_index += 1
                pass
            elif nn_weight == 0 and random_point_weight != 0:  
                # proportion = 1
                X_new_zhou = np.array(random_point_data)
                X_new_1 = np.vstack((X_new_zhou, X_new_1))
            elif nn_weight != 0 and random_point_weight == 0:  
                # proportion = 0
                X_new_zhou = np.array(nn_point)

                X_new_1 = np.vstack((X_new_zhou, X_new_1))

    
    for mm in range(m + delet_index):
        if not isinstance(nn_data, list):nn_data = nn_data.tolist()       

        nn_points = random.choice(nn_data)  

        nn_point_index = nn_data.index(nn_points)
        nn_point_weight = new_n_maj[nn_point_index]  

        num = nn_num[nn_point_index].tolist()  
        random_point_index = random.choice(num)  
        random_point_weights = new_n_maj[random_point_index]  
        # print('num:\t', num, 'random_point_index:\t', random_point_index,'random_point_weight:\t',random_point_weight)
        random_points_data = nn_data[random_point_index]

        # if nn_point_weight!=0 and random_point_weights!=0:
        #     proportion = (random_point_weights/(nn_point_weight+random_point_weights))  
        #     X_new_zhou = np.array(nn_points) + (np.array(random_points_data)-np.array(nn_points)) * proportion*round(random.uniform(0,1),len(str(n_samples)))

        if nn_point_weight != 0 and random_point_weights != 0:  
            proportion = (random_point_weights / (nn_point_weight + random_point_weights))  
            if nn_point_weight >= random_point_weights:
                X_new_zhou = np.array(nn_points) + (
                            np.array(random_points_data) - np.array(nn_points)) * proportion * round(random.uniform(0, 1),
                                                                                                len(str(n_samples)))
            elif nn_point_weight < random_point_weights:
                X_new_zhou = np.array(random_points_data) + (np.array(nn_points) - np.array(random_points_data)) * (
                            1 - proportion) * round(random.uniform(0, 1), len(str(n_samples)))
            X_new_1 = np.vstack((X_new_zhou, X_new_1))
        elif (nn_point_weight + random_point_weights) == 0:  
            # proportion = 0
            # X_new_zhou = np.array(nn_points)
            pass
        elif nn_point_weight == 0 and random_point_weights != 0:
            X_new_zhou = np.array(random_points_data)
            X_new_1 = np.vstack((X_new_zhou, X_new_1))
        elif nn_point_weight != 0 and random_point_weights == 0:
            X_new_zhou = np.array(nn_points)

            X_new_1 = np.vstack((X_new_zhou, X_new_1))

    X_new_1 = np.delete(X_new_1, -1, 0)
    print(X_new_1.shape)
    return X_new_1.astype(X.dtype)


def generate_samples_zhou_borderline(X, nn_data, nn_num, rows, cols, steps, n_samples,
                                     new_n_maj, danger_and_safe):
    '''
zhou
# n = (majority class - minority class) //minority class takes integer number of points to be interpolated
# m = (majority class - minority class)%minority class takes remainder and the rest is used for random
#Traverse the boundary and safe points, find the corresponding KNN points according to the neighbor point index matrix, randomly extract with replacement N times, subtract the coordinates of the two points, calculate the distance according to the weight ratio to interpolate,
#The remaining m points are used for random extraction without replacement, calculate the weight ratio of the extracted points and the neighbor points, and each point is only interpolated once
        
    '''
    danger_data = _safe_indexing(nn_data, danger_and_safe)  # 边界点的横纵坐标

    n = n_samples // len(danger_data)
    m = n_samples % len(danger_data)
    # print('n,m,n_samples',n,m,n_samples)
    weidu = X.shape[1]
    X_new_1 = np.zeros(weidu)


    delete_index = 0
    for nn in range(len(danger_data)):  # 遍历每个边界点

        num = nn_num[nn].tolist()  # ndarray，每个点的近邻点索引矩阵
        if not isinstance(nn_data, list):nn_data = nn_data.tolist()

        nn_point = nn_data[np.flatnonzero(danger_and_safe)[nn]]  # 当前选择的母节点
        nn_weight = new_n_maj[np.flatnonzero(danger_and_safe)[nn]]  # 当前亩节点的权重

        for i in range(n):  # 随机有放回的抽取N次近邻点
            # random_point = random.choice(num)  # 随机点在  矩阵中的索引
            random_point = num[i%len(num)]           #TODO按距离由进到选抽取近邻点

            # print('random_point:\t',random_point)
            # random_point_index = num.index(random_point)        #随机点在近邻矩阵中的索引
            random_point_weight = new_n_maj[random_point]  # 随机点权重
            random_point_data = nn_data[random_point]  # 随机点的横纵坐标
            # print(random_point, num.index(random_point),random_point_data)

            # if nn_weight!=0 and random_point_weight!=0:
            #     proportion = (random_point_weight/(nn_weight+random_point_weight))#权重比例
            #     X_new_zhou = np.array(nn_point) + (np.array(random_point_data) - np.array(nn_point))*proportion*round(random.uniform(0,1),len(str(n_samples)))
            if nn_weight != 0 and random_point_weight != 0:  # 两个都不是噪声点
                proportion = (random_point_weight / (nn_weight + random_point_weight))  # 权重比例
                if nn_weight >= random_point_weight:
                    X_new_zhou = np.array(nn_point) + (
                                np.array(random_point_data) - np.array(nn_point)) * proportion * round(
                        random.uniform(0, 1), len(str(n_samples)))
                elif nn_weight < random_point_weight:
                    X_new_zhou = np.array(random_point_data) + (np.array(nn_point) - np.array(random_point_data)) * (
                                1 - proportion) * round(random.uniform(0, 1), len(str(n_samples)))

            elif (nn_weight + random_point_weight) == 0:  # 如果母点和随机点权重都是0（两个点都是噪声点）
                # X_new_zhou = np.array(nn_point)
                delete_index += 1
                pass
            elif nn_weight == 0 and random_point_weight != 0:
                X_new_zhou = np.array(random_point_data)
            elif nn_weight != 0 and random_point_weight == 0:  # 母点不是噪声点，随机点是噪声点
                X_new_zhou = np.array(nn_point)

                # random_float = round(random.uniform(0,1),len(str(n_samples)))
            # X_new_zhou = np.array(nn_point) + (np.array(random_point_data) - np.array(nn_point))*proportion*round(random.uniform(0,1),len(str(n_samples)))
            # X_new_zhou = np.array(nn_point) + (np.array(random_point_data) - np.array(nn_point))*proportion

            X_new_1 = np.vstack((X_new_zhou, X_new_1))

    # 随机不放回抽取m个点
    for mm in range(m + delete_index):
        if not isinstance(danger_data, list):danger_data = danger_data.tolist()

        nn_points = random.choice(danger_data)  # 有放回地随机抽取一个边界点
        # print('nn_points:\t',nn_points)
        nn_point_index = danger_data.index(nn_points)
        # print('nn_point_index:\t',nn_point_index)
        nn_point_weight = new_n_maj[np.flatnonzero(danger_and_safe)[nn]]  # 抽取点的权重

        num = nn_num[nn_point_index].tolist()  # 抽取点的近邻点列表
        random_point_index = random.choice(num)  # 随机近邻点的索引
        random_point_weights = new_n_maj[random_point_index]  # 随机近邻点的权重
        # print('num:\t', num, 'random_point_index:\t', random_point_index,'random_point_weight:\t',random_point_weight)
        random_points_data = nn_data[random_point_index]

        # if nn_point_weight!=0 and random_point_weights!=0:
        #     proportion = (random_point_weights/(nn_point_weight+random_point_weights))#权重比例
        #     X_new_zhou = np.array(nn_points) + (np.array(random_points_data)-np.array(nn_points)) * proportion*round(random.uniform(0,1),len(str(n_samples)))
        if nn_point_weight != 0 and random_point_weights != 0:  # 两个都不是噪声点
            proportion = (random_point_weights / (nn_point_weight + random_point_weights))  # 权重比例
            if nn_point_weight >= random_point_weights:
                X_new_zhou = np.array(nn_points) + (
                            np.array(random_points_data) - np.array(nn_points)) * proportion * round(random.uniform(0, 1),
                                                                                                len(str(n_samples)))
            elif nn_point_weight < random_point_weights:
                X_new_zhou = np.array(random_points_data) + (np.array(nn_points) - np.array(random_points_data)) * (
                            1 - proportion) * round(random.uniform(0, 1), len(str(n_samples)))


        elif (nn_point_weight + random_point_weights) == 0:  # 如果母点和随机点权重都是0（两个点都是噪声点）
            # proportion = 0
            # X_new_zhou = np.array(nn_points)
            pass
        elif nn_point_weight == 0 and random_point_weights != 0:
            X_new_zhou = np.array(random_points_data)
        elif nn_point_weight != 0 and random_point_weights == 0:
            X_new_zhou = np.array(nn_points)

        X_new_1 = np.vstack((X_new_zhou, X_new_1))

    X_new_1 = np.delete(X_new_1, -1, 0)
    return X_new_1.astype(X.dtype)


def generate_samples_zhou_svm_smote( X, nn_data, nn_num, rows, cols, steps,n_samples,
                new_n_maj,danger_and_safe,mother_point):

    '''
        zhou
        # n = （多数类-少数类）//少数类    取整数     每个点要插的数量
        # m = （多数类-少数类）%少数类    取余数         多余的用来随机
        #遍历边界和安全点，根据近邻点索引矩阵找到对应的KNN个点，随机有放回抽取N次，两点坐标相减，按权重比算距离来插值，
        #剩下的m个点用来随机不放回抽点，计算抽到的点和近邻点的权重比,每个点只插一次
    '''
    # print('danger_and_safe：\t',len(np.flatnonzero(danger_and_safe)))
    lenth = len(np.flatnonzero(danger_and_safe))
    # print(danger_and_safe)
    # print(new_n_maj)
    # print(lenth,nn_data)
    # print(mother_point,len(mother_point))

    mother_points = nn_data[mother_point]       #mother_point是索引，mother_points是点
    # print(len(nn_data),type(nn_data),mother_points)

    n = n_samples // lenth
    m = n_samples %  lenth

    # print('n,m,n_samples',n,m,n_samples)
    weidu = X.shape[1]
    # print('维度:\t',weidu)
    X_new_1 = np.zeros(weidu)
    # print(X_new_1.shape,'\n',X_new_1)

    delete_index=0
    '''每个点需要插的n个点'''
    for nn in range(lenth):      #每个点的横纵坐标值

        num = nn_num[nn].tolist()            #ndarray，每个点的近邻点索引矩阵
        if  not isinstance(nn_data,list): nn_data = nn_data.tolist()

        nn_point = mother_points[nn]      #当前选择的母节点的横纵坐标
        nn_weight = new_n_maj[mother_point[nn]]       #当前母节点的权重
        # print('母节点:\t',nn_point,num,nn_weight)

        for i in range(n):            #随机有放回的抽取N次近邻点
            # random_point = random.choice(num)               #随机点在  nns中的索引
            random_point = num[i%len(num)]           #TODO按距离由进到选抽取近邻点

            # print('randow_point_1:\t',random_point)
            # random_point_index = num.index(random_point)        #随机点在近邻矩阵中的索引
            random_point_weight = new_n_maj[random_point]           #随机点权重
            random_point_data = nn_data[random_point]               #随机点的横纵坐标

            # if nn_weight!= 0 and random_point_weight!=0:        #两个都不是噪声点
            #     proportion = (random_point_weight/(nn_weight+random_point_weight))#权重比例
            #     X_new_zhou = np.array(nn_point) + (np.array(random_point_data) - np.array(nn_point))*proportion*round(random.uniform(0,1),len(str(n_samples)))
            if nn_weight!= 0 and random_point_weight!=0:        #两个都不是噪声点
                proportion = (random_point_weight/(nn_weight+random_point_weight))#权重比例
                if nn_weight >= random_point_weight:
                    X_new_zhou = np.array(nn_point) + (np.array(random_point_data) - np.array(nn_point))*proportion*round(random.uniform(0,1),len(str(n_samples)))  
                elif nn_weight < random_point_weight:
                    X_new_zhou = np.array(random_point_data) + (np.array(nn_point)-np.array(random_point_data))*(1-proportion)*round(random.uniform(0,1),len(str(n_samples)))
            
            elif (nn_weight+random_point_weight)==0:       #如果母点和随机点权重都是0（两个点都是噪声点）
                # X_new_zhou = np.array(nn_point)
                delete_index+=1
                pass
            elif nn_weight==0 and random_point_weight !=0:  #母点是噪声点，随机点不是噪声点
                X_new_zhou = np.array(random_point_data)            
            elif nn_weight!=0 and random_point_weight ==0:  #母点不是噪声点，随机点是噪声点
                X_new_zhou = np.array(nn_point)

            X_new_1 = np.vstack((X_new_zhou,X_new_1))
            

    '''随机不放回抽取m个点'''
    for mm in range(m+delete_index):

        nn_point_index = random.choice(mother_point)        #母点的索引
        nn_point_weight = new_n_maj[nn_point_index]         #母点的权重
        nn_points = nn_data[nn_point_index]                  #母点的横纵坐标
        a = np.where(mother_point == nn_point_index)[0][0]
        # print('nn_point:\t',nn_point_index,nn_point,nn_point_weight)

        num = nn_num[a].tolist()       #抽取点的近邻点列表
        random_point_index = random.choice(num)
        random_point_weights = new_n_maj[random_point_index]
        # print('num:\t', num, 'random_point_index:\t', random_point_index,'random_point_weight:\t',random_point_weight)
        random_points_data = nn_data[random_point_index]

        # if nn_point_weight!=0 and random_point_weights!=0:
        #     proportion = (random_point_weights/(nn_point_weight+random_point_weights))#权重比例
        #     X_new_zhou = np.array(nn_points) + (np.array(random_points_data)-np.array(nn_points)) * proportion*round(random.uniform(0,1),len(str(n_samples)))
        if nn_point_weight!=0 and random_point_weights!=0:       #两个都不是噪声点
                proportion = (random_point_weight/(nn_weight+random_point_weight))#权重比例
                if nn_point_weight >= random_point_weights:
                    X_new_zhou = np.array(nn_point) + (np.array(random_point_data) - np.array(nn_point))*proportion*round(random.uniform(0,1),len(str(n_samples)))  
                elif nn_point_weight < random_point_weights:
                    X_new_zhou = np.array(random_point_data) + (np.array(nn_point)-np.array(random_point_data))*(1-proportion)*round(random.uniform(0,1),len(str(n_samples)))
            
        elif (nn_point_weight+random_point_weights)==0:       #如果母点和随机点权重都是0（两个点都是噪声点）
            # proportion = 0
            # X_new_zhou = np.array(nn_points)
            pass
        elif nn_point_weight==0 and random_point_weights !=0:
            X_new_zhou = np.array(random_points_data)
        elif nn_point_weight !=0 and random_point_weights ==0:
            X_new_zhou = np.array(nn_points)

        X_new_1 = np.vstack((X_new_zhou, X_new_1))

    X_new_1 = np.delete(X_new_1, -1, 0)
    return X_new_1.astype(X.dtype)
#-----------------------------------------------------------------------------------



class BaseSMOTE(BaseOverSampler):
    """Base class for the different SMOTE algorithms."""

    def __init__(
        self,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=10,
        n_jobs=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs


    def _validate_estimator(self):
        """Check the NN estimators shared across the different SMOTE
        algorithms.
        """
        self.nn_k_ = check_neighbors_object(
            "k_neighbors", self.k_neighbors, additional_neighbor=1,
            # p=1 #TODO,改距离度量
        )
        # print('self.nn_k_.p**********************:\t',self.nn_k_.p)
        self.nn_k_.set_params(**{"n_jobs": self.n_jobs})


    def _make_samples(
        self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0
    ):
        """A support function that returns artificial samples constructed along
        the line connecting nearest neighbours.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Points from which the points will be created.

        y_dtype : dtype
            The data type of the targets.

        y_type : str or int
            The minority target value, just so the function can return the
            target values for the synthetic variables with correct length in
            a clear format.

        nn_data : ndarray of shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used

        nn_num : ndarray of shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

        n_samples : int
            The number of samples to generate.

        step_size : float, default=1.0
            The step size to create samples.

        Returns
        -------
        X_new : {ndarray, sparse matrix} of shape (n_samples_new, n_features)
            Synthetically generated samples.

        y_new : ndarray of shape (n_samples_new,)
            Target values for synthetic samples.
        """
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(
            low=0, high=nn_num.size, size=n_samples
        )

        # np.newaxis for backwards compatability with random_state
        steps = step_size * random_state.uniform(size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])

        X_new = self._generate_samples(X, nn_data, nn_num, rows, cols, steps)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new


    def _make_samples_zhou(
        self, X, y_dtype, y_type, nn_data, nn_num, n_samples, 
                    new_n_maj,danger_and_safe=None,step_size=1.0,
    ):
        """A support function that returns artificial samples constructed along
        the line connecting nearest neighbours.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Points from which the points will be created.

        y_dtype : dtype
            The data type of the targets.

        y_type : str or int
            The minority target value, just so the function can return the
            target values for the synthetic variables with correct length in
            a clear format.

        nn_data : ndarray of shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used

        nn_num : ndarray of shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

        n_samples : int
            The number of samples to generate.

        step_size : float, default=1.0
            The step size to create samples.

        Returns
        -------
        X_new : {ndarray, sparse matrix} of shape (n_samples_new, n_features)
            Synthetically generated samples.

        y_new : ndarray of shape (n_samples_new,)
            Target values for synthetic samples.
        """


        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(
            low=0, high=nn_num.size, size=n_samples
        )
        # print('samples_indices:\t',samples_indices,len(samples_indices))


        # np.newaxis for backwards compatability with random_state
        steps = step_size * random_state.uniform(size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])       #相当于除法取整
        cols = np.mod(samples_indices, nn_num.shape[1])
        # print('steps:\n',steps,len(steps),'\nrows:\n',rows,len(rows),'\ncols:\n',cols,len(cols))


        X_new = self._generate_samples_zhou(X, nn_data, nn_num, rows, cols, steps,n_samples,
                    new_n_maj,danger_and_safe)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)


        return X_new, y_new


    def _generate_samples_zhou(self, X, nn_data, nn_num, rows, cols, steps,n_samples,
                    new_n_maj,danger_and_safe):

        '''原始方法'''
        diffs = nn_data[nn_num[rows, cols]] - X[rows]


        if sparse.issparse(X):              #判断是否是稀疏矩阵
            sparse_func = type(X).__name__
            steps = getattr(sparse, sparse_func)(steps)
            X_new = X[rows] + steps.multiply(diffs)
        else:
            X_new = X[rows] + steps * diffs




        '''
zhou
# n = (majority class - minority class) //minority class takes integer number of points to be interpolated
# m = (majority class - minority class)%minority class takes remainder and the rest is used for random
#Traverse the boundary and safe points, find the corresponding KNN points according to the neighbor point index matrix, randomly extract with replacement N times, subtract the coordinates of the two points, calculate the distance according to the weight ratio to interpolate,
#The remaining m points are used for random extraction without replacement, calculate the weight ratio of the extracted points and the neighbor points, and each point is only interpolated once
        '''

        if danger_and_safe is not None:
            nn_data = _safe_indexing(nn_data, danger_and_safe)  # 边界点和安全点
        else:pass

        # print('nn_data:\t',len(nn_data))
        n = n_samples // len(nn_data)
        m = n_samples %  len(nn_data)
        # print('n,m,n_samples',n,m,n_samples)
        weidu = X.shape[1]
        # print('维度:\t',weidu)
        X_new_1 = np.zeros(weidu)
        # print(X_new_1.shape,'\n',X_new_1)

        for nn in range(len(nn_data)):      

            if nn==1:
                # break
                pass

            num = nn_num[nn]            
            num = num.tolist()
            if  isinstance(nn_data,list): pass
            else:nn_data = nn_data.tolist()
            nn_point = nn_data[nn]
            # print(nn_point)
            # print(num,type(num))
            # print(num)
            for i in range(n):            
                # print(nn,i)
                random_point = random.choice(num)               
                # print('randow_point_1:\t',random_point)
                # random_point_index = num.index(random_point)        
                nn_weight = new_n_maj[nn]
                random_point_weight = new_n_maj[random_point]           
                random_point_data = nn_data[random_point]               
                # print(random_point, num.index(random_point),random_point_data)

                proportion = (random_point_weight/(nn_weight+random_point_weight))     
                X_new_zhou = np.array(nn_point) + (np.array(random_point_data) - np.array(nn_point))*proportion
                X_new_1 = np.vstack((X_new_zhou,X_new_1))





       #Randomly extract m points without replacement
        for mm in range(m):
            if isinstance(nn_data, list):pass
            else:nn_data = nn_data.tolist()
            nn_point = random.choice(nn_data)                       
            # nn_data.remove(nn_point)                                
            nn_point_index = nn_data.index(nn_point)
            nn_point_weight = new_n_maj[nn_point_index]         

            num = nn_num[nn_point_index].tolist()       
            random_point_index = random.choice(num)
            random_point_weight = new_n_maj[random_point_index]
            # print('num:\t', num, 'random_point_index:\t', random_point_index,'random_point_weight:\t',random_point_weight)
            random_point_data = nn_data[random_point_index]
            proportion = (random_point_weight / (nn_point_weight + random_point_weight))

            X_new_zhou = np.array(nn_point) + (np.array(random_point_data)-np.array(nn_point)) * proportion
            X_new_1 = np.vstack((X_new_zhou, X_new_1))


        X_new_1 = np.delete(X_new_1, -1, 0)


        return X_new_1.astype(X.dtype)
        # return X_new.astype(X.dtype)


    def _generate_samples(self, X, nn_data, nn_num, rows, cols, steps):
        r"""Generate a synthetic sample.

        The rule for the generation is:

        .. math::
           \mathbf{s_{s}} = \mathbf{s_{i}} + \mathcal{u}(0, 1) \times
           (\mathbf{s_{i}} - \mathbf{s_{nn}}) \,

        where \mathbf{s_{s}} is the new synthetic samples, \mathbf{s_{i}} is
        the current sample, \mathbf{s_{nn}} is a randomly selected neighbors of
        \mathbf{s_{i}} and \mathcal{u}(0, 1) is a random number between [0, 1).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Points from which the points will be created.

        nn_data : ndarray of shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used.

        nn_num : ndarray of shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

        rows : ndarray of shape (n_samples,), dtype=int
            Indices pointing at feature vector in X which will be used
            as a base for creating new samples.

        cols : ndarray of shape (n_samples,), dtype=int
            Indices pointing at which nearest neighbor of base feature vector
            will be used when creating new samples.

        steps : ndarray of shape (n_samples,), dtype=float
            Step sizes for new samples.

        Returns
        -------
        X_new : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Synthetically generated samples.
        """
        diffs = nn_data[nn_num[rows, cols]] - X[rows]       


        if sparse.issparse(X):              
            sparse_func = type(X).__name__
            steps = getattr(sparse, sparse_func)(steps)
            X_new = X[rows] + steps.multiply(diffs)
        else:
            # print(len(X[rows]),len(steps*diffs),'\n')     # 500,500
            X_new = X[rows] + steps * diffs



        return X_new.astype(X.dtype)

    def _in_danger_noise(
        self, nn_estimator, samples, target_class, y, kind="danger"
    ):
        """Estimate if a set of sample are in danger or noise.

        Used by BorderlineSMOTE and SVMSMOTE.

        Parameters
        ----------
        nn_estimator : estimator
            An estimator that inherits from
            :class:`sklearn.neighbors.base.KNeighborsMixin` use to determine if
            a sample is in danger/noise.

        samples : {array-like, sparse matrix} of shape (n_samples, n_features)
            The samples to check if either they are in danger or not.

        target_class : int or str
            The target corresponding class being over-sampled.

        y : array-like of shape (n_samples,)
            The true label in order to check the neighbour labels.

        kind : {'danger', 'noise'}, default='danger'
            The type of classification to use. Can be either:

            - If 'danger', check if samples are in danger,
            - If 'noise', check if samples are noise.

        Returns
        -------
        output : ndarray of shape (n_samples,)
            A boolean array where True refer to samples in danger or noise.
        """
        x = nn_estimator.kneighbors(samples, return_distance=False)[:, 1:]
        # print('X:\t',x)
        nn_label = (y[x] != target_class).astype(int)
        n_maj = np.sum(nn_label, axis=1)

        # print('The number of majority class points in the K nearest neighbors of each minority class point:\t',n_maj,len(n_maj))
        # print('nn_estimator.n_neighbors:\t',nn_estimator.n_neighbors)     # ==self.k_neigbors

        if kind == "danger":
            # Samples are in danger for m/2 <= m' < m
            return np.bitwise_and(          
                n_maj >= (nn_estimator.n_neighbors - 1) / 2,
                n_maj < nn_estimator.n_neighbors - 1,
            ),n_maj

        elif kind == "noise":
            # Samples are noise for m = m'
            return n_maj == nn_estimator.n_neighbors - 1,n_maj
        else:
            raise NotImplementedError


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class BorderlineSMOTE(BaseSMOTE):
    """
        k_neighbors : int or object, default=5
            If ``int``, number of nearest neighbours to used to construct synthetic
            samples.  If object, an estimator that inherits from
            :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
            find the k_neighbors.
        {n_jobs}
        m_neighbors : int or object, default=10
            If int, number of nearest neighbours to use to determine if a minority
            sample is in danger. If object, an estimator that inherits
            from :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used
            to find the m_neighbors.
    """

    '''
nn_m is used to determine boundary points, dangerous points, and safe points, m_neighbors=10, inherited from the Base class
nn_k is used to find the k nearest neighbor points in the minority class points k_neighbors=5,
    '''
    def __init__(
        self,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,      
        n_jobs=None,
        m_neighbors=5,     #TODO
        kind="borderline-1",
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.m_neighbors = m_neighbors
        self.k_neighbors = k_neighbors
        self.kind = kind


    def _validate_estimator(self):
        super()._validate_estimator()
        self.nn_m_ = check_neighbors_object(
            "m_neighbors", self.m_neighbors, additional_neighbor=1     #TODO
        )
        self.nn_m_.set_params(**{"n_jobs": self.n_jobs})
        if self.kind not in ("borderline-1", "borderline-2",'weight-borderline-smote'):
            raise ValueError(
                'The possible "kind" of algorithm are '
                '"borderline-1" and "borderline-2".'
                "Got {} instead.".format(self.kind)
            )


    def _fit_resample(self, X, y):
        self._validate_estimator()
        X_resampled = X.copy()
        y_resampled = y.copy()
        
        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:continue

            target_class_indices = np.flatnonzero(y == class_sample)    
            X_class = _safe_indexing(X, target_class_indices)       

            self.nn_m_.fit(X)           
            danger_index ,n_maj= self._in_danger_noise(
                self.nn_m_, X_class, class_sample, y, kind="danger"
            )
            if not any(danger_index):continue

            noise , n_maj= self._in_danger_noise(
                self.nn_m_, X_class, class_sample, y, kind="noise"
            )
            if not any(noise):print('没有噪声')

            danger_and_safe = (noise==False)                #bool

            ''',计算每个少数类点的权重'''
            def conut_weight(n_maj,noise_index=None):
                if isinstance(noise_index,np.ndarray):
                    n_maj = np.delete(n_maj,noise_index)            
                new_n_maj = [round((1-i/self.m_neighbors),2) for i in n_maj]        
                return new_n_maj

            new_n_maj = conut_weight(n_maj=n_maj)

# Returns the index and distance of the K nearest neighbors of the safe point and the boundary point in the minority points. nn_k is to find the k nearest neighbors in the minority class points
# self.nn_k_.fit(_safe_indexing(X_class, danger_and_safe))
# nns= self.nn_k_.kneighbors(
# _safe_indexing(X_class, danger_and_safe), return_distance=False
# )[:,1:]

            self.nn_k_.fit(X_class)
            nns = self.nn_k_.kneighbors(
                _safe_indexing(X_class, danger_index), return_distance=False
            )[:, 1:]

            # divergence between borderline-1 and borderline-2
            if self.kind == 'weight-borderline-smote':
                X_new, y_new = make_samples_zhou(
                    _safe_indexing(X_class, danger_index),          
                    # _safe_indexing(X_class,danger_and_safe),                
                    y.dtype,            
                    class_sample,       
                    X_class,                               
                    nns,                
                    n_samples,                           
                    new_n_maj,
                    danger_index,
                    1.0,
                    kind='borderline'
                )

                
                if sparse.issparse(X_new): X_resampled = sparse.vstack([X_resampled, X_new])
                else:X_resampled = np.vstack((X_resampled, X_new))    
                y_resampled = np.hstack((y_resampled, y_new))

            elif self.kind == "borderline-1":
                
                # Create synthetic samples for borderline points.
                nns = self.nn_k_.kneighbors(
                    _safe_indexing(X_class, danger_index), return_distance=False
                )[:, 1:]

                X_new, y_new = self._make_samples(
                    _safe_indexing(X_class, danger_index),
                    y.dtype,            
                    class_sample,       
                    X_class,            
                    nns,                
                    n_samples,          
                )
                if sparse.issparse(X_new):
                    X_resampled = sparse.vstack([X_resampled, X_new])
                else:
                    X_resampled = np.vstack((X_resampled, X_new))
                y_resampled = np.hstack((y_resampled, y_new))

            elif self.kind == "borderline-2":
                nns = self.nn_k_.kneighbors(
                    _safe_indexing(X_class, danger_index), return_distance=False
                )[:, 1:]
                random_state = check_random_state(self.random_state)
                fractions = random_state.beta(10, 10)

                # only minority
                X_new_1, y_new_1 = self._make_samples(
                    _safe_indexing(X_class, danger_index),
                    y.dtype,
                    class_sample,
                    X_class,
                    nns,
                    int(fractions * (n_samples + 1)),
                    step_size=1.0,
                )

                # we use a one-vs-rest policy to handle the multiclass in which
                # new samples will be created considering not only the majority
                # class but all over classes.
                X_new_2, y_new_2 = self._make_samples(
                    _safe_indexing(X_class, danger_index),
                    y.dtype,
                    class_sample,
                    _safe_indexing(X, np.flatnonzero(y != class_sample)),
                    nns,
                    int((1 - fractions) * n_samples),
                    step_size=0.5,
                )

                if sparse.issparse(X_resampled):
                    X_resampled = sparse.vstack(
                        [X_resampled, X_new_1, X_new_2]
                    )
                else:
                    X_resampled = np.vstack((X_resampled, X_new_1, X_new_2))
                y_resampled = np.hstack((y_resampled, y_new_1, y_new_2))

            return  X_resampled,y_resampled




@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class SVMSMOTE(BaseSMOTE):
    """Over-sampling using SVM-SMOTE.

    Variant of SMOTE algorithm which use an SVM algorithm to detect sample to
    use for generating new synthetic samples as proposed in [2]_.

    Read more in the :ref:`User Guide <smote_adasyn>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    k_neighbors : int or object, default=5
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples.  If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.

    {n_jobs}

    m_neighbors : int or object, default=10
        If int, number of nearest neighbours to use to determine if a minority
        sample is in danger. If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the m_neighbors.

    svm_estimator : object, default=SVC()
        A parametrized :class:`sklearn.svm.SVC` classifier can be passed.

    out_step : float, default=0.5
        Step size when extrapolating.

    See Also
    --------
    SMOTE : Over-sample using SMOTE.

    SMOTENC : Over-sample using SMOTE for continuous and categorical features.

    BorderlineSMOTE : Over-sample using Borderline-SMOTE.

    ADASYN : Over-sample using ADASYN.

    KMeansSMOTE : Over-sample applying a clustering before to oversample using
        SMOTE.

    Notes
    -----
    See the original papers: [2]_ for more details.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE:
       synthetic minority over-sampling technique," Journal of artificial
       intelligence research, 321-357, 2002.

    .. [2] H. M. Nguyen, E. W. Cooper, K. Kamei, "Borderline over-sampling for
       imbalanced data classification," International Journal of Knowledge
       Engineering and Soft Data Paradigms, 3(1), pp.4-21, 2009.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.over_sampling import \
SVMSMOTE # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> sm = SVMSMOTE(random_state=42)
    >>> X_res, y_res = sm.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 900, 1: 900}})
    """

    def __init__(
        self,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
        m_neighbors=5,
        svm_estimator=None,
        out_step=0.5,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.m_neighbors = m_neighbors
        self.svm_estimator = svm_estimator
        self.out_step = out_step

    def _validate_estimator(self):
        super()._validate_estimator()
        self.nn_m_ = check_neighbors_object(
            "m_neighbors",self.m_neighbors, additional_neighbor=1     #TODO
        )
        self.nn_m_.set_params(**{"n_jobs": self.n_jobs})

        if self.svm_estimator is None:
            self.svm_estimator_ = SVC(
                gamma="scale", random_state=self.random_state
            )
        elif isinstance(self.svm_estimator, SVC):
            self.svm_estimator_ = clone(self.svm_estimator)
        else:
            raise_isinstance_error("svm_estimator", [SVC], self.svm_estimator)


    def _fit_resample(self, X, y):
        self._validate_estimator()
        random_state = check_random_state(self.random_state)
        X_resampled = X.copy()
        y_resampled = y.copy()

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            # print('target_class_indices:\n',target_class_indices,len(target_class_indices),type(target_class_indices))
            X_class = _safe_indexing(X, target_class_indices)


            
            self.svm_estimator_.fit(X, y)
            support_index = self.svm_estimator_.support_[               #np.ndarray
                y[self.svm_estimator_.support_] == class_sample
            ]           
            support_vector = _safe_indexing(X, support_index)       
            # print('Sn：\t',len(support_vector))
            # print('support_index:\t',support_index,len(support_index),type(support_index))
            

            
            self.nn_m_.fit(X)           
            noise_bool = self._in_danger_noise(
                self.nn_m_, support_vector, class_sample, y, kind="noise"
            )[0]            #TODO,
            # print('noise_bool:\t',noise_bool,len(noise_bool))
            noise_index = np.flatnonzero(noise_bool)
            # print('noise_index:\t',noise_index)
            support_index = np.delete(support_index,noise_index)
            # print(':\t',support_index,len(support_index))



            
            support_vector = _safe_indexing(
                support_vector, np.flatnonzero(np.logical_not(noise_bool))
                # support_vector, support_index
            )       
            # print('：\t',len(support_vector))
            


            
            danger__bool,n__maj = self._in_danger_noise(
                self.nn_m_, X_class, class_sample, y, kind="danger"
            )

            
            danger_bool,n_maj = self._in_danger_noise(
                self.nn_m_, support_vector, class_sample, y, kind="danger"
            )
            safety_bool = np.logical_not(danger_bool)       


            danger_index = np.delete(support_index,np.flatnonzero(safety_bool))
            safe_index = np.delete(support_index,np.flatnonzero(danger_bool))

            # print(':\t',danger_index)
            # print(':\t',safe_index)
            danger_list = []
            safe_list = []
            for i in danger_index:
                # import numpy as np
                ii = np.where(target_class_indices == i)
                # print(ii[0][0])
                danger_list.append(ii[0][0])

            for i in safe_index:
                # import numpy as np
                ii = np.where(target_class_indices == i)
                # print(ii[0][0])
                safe_list.append(ii[0][0])

            # print(danger_list,safe_list)




            safe_vector = _safe_indexing(
                support_vector, np.flatnonzero(np.logical_not(danger_bool)
            ))
            danger_vector = _safe_indexing(
                support_vector, np.flatnonzero(danger_bool)
            )


            self.nn_k_.fit(X_class)     
            fractions = random_state.beta(10, 10)
            n_generated_samples = int(fractions * (n_samples + 1))
            # print('：\t',np.count_nonzero(danger_bool))
            # print('：\t',np.count_nonzero(safety_bool))


            def conut_weight(n_maj,index):
                # print(type(n_maj),type(noise_index))      #ndarray
                new_n_maj = [round((1-i/self.m_neighbors),2) for i in n_maj]
                return new_n_maj
            new_n_maj = conut_weight(n_maj=n__maj,index=None)
            # print('：\t',n_generated_samples,'\:\t',n_samples - n_generated_samples,)
            # print(':\t',new_n_maj,len(new_n_maj))




            
            if np.count_nonzero(danger_bool) > 0:
                # print('a')
                
                nns = self.nn_k_.kneighbors(
                    _safe_indexing(support_vector, np.flatnonzero(danger_bool)),
                    return_distance=False,
                )[:, 1:]

                # print(nns)        

                X_new_1, y_new_1 = make_samples_zhou(
                    X=_safe_indexing(support_vector, np.flatnonzero(danger_bool)),
                    y_dtype=y.dtype,
                    y_type=class_sample,
                    nn_data=X_class,                
                    nn_num=nns,                     
                    n_samples=n_generated_samples,      
                    # new_n_maj=danger_n_maj,
                    new_n_maj=new_n_maj,
                    danger_and_safe=danger_bool,
                    kind='svm',
                    mother_point = np.array(danger_list),
    
                )


            
            if np.count_nonzero(safety_bool) > 0:
                # print('b')
                
                nns = self.nn_k_.kneighbors(
                    _safe_indexing(support_vector, np.flatnonzero(safety_bool)),
                    return_distance=False,
                )[:, 1:]

                X_new_2, y_new_2 = make_samples_zhou(
                    X=_safe_indexing(support_vector, np.flatnonzero(safety_bool)),
                    y_dtype=y.dtype,
                    y_type=class_sample,
                    nn_data=X_class,
                    nn_num=nns,
                    n_samples=n_samples - n_generated_samples,
                    # new_n_maj=safety_n_maj,
                    new_n_maj=new_n_maj,
                    # step_size=-self.out_step,
                    danger_and_safe=safety_bool,
                    kind='svm',
                    mother_point=np.array(safe_list),
                )


            if (
                np.count_nonzero(danger_bool) > 0
                and np.count_nonzero(safety_bool) > 0
            ):
                # print('c')
                if sparse.issparse(X_resampled):
                    X_resampled = sparse.vstack(
                        [X_resampled, X_new_1, X_new_2]
                    )
                else:
                    X_resampled = np.vstack((X_resampled, X_new_1, X_new_2))
                y_resampled = np.concatenate(
                    (y_resampled, y_new_1, y_new_2), axis=0
                )
            elif np.count_nonzero(danger_bool) == 0:
                if sparse.issparse(X_resampled):
                    X_resampled = sparse.vstack([X_resampled, X_new_2])
                else:
                    X_resampled = np.vstack((X_resampled, X_new_2))
                y_resampled = np.concatenate((y_resampled, y_new_2), axis=0)
            elif np.count_nonzero(safety_bool) == 0:
                if sparse.issparse(X_resampled):
                    X_resampled = sparse.vstack([X_resampled, X_new_1])
                else:
                    X_resampled = np.vstack((X_resampled, X_new_1))
                y_resampled = np.concatenate((y_resampled, y_new_1), axis=0)

        return X_resampled, y_resampled


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class SMOTE(BaseSMOTE):
    """
    k_neighbors : int or object, default=5
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples.  If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.

    {n_jobs}

    See Also
    --------
    SMOTENC : Over-sample using SMOTE for continuous and categorical features.

    BorderlineSMOTE : Over-sample using the borderline-SMOTE variant.

    SVMSMOTE : Over-sample using the SVM-SMOTE variant.

    ADASYN : Over-sample using ADASYN.

    KMeansSMOTE : Over-sample applying a clustering before to oversample using
        SMOTE.

    Notes
    -----
    See the original papers: [1]_ for more details.
    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.


    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.over_sampling import \
    SMOTE # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> sm = SMOTE(random_state=42)
    >>> X_res, y_res = sm.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 900, 1: 900}})
    """

    def __init__(
        self,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
        m_neighbors=5
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.m_neighbors = m_neighbors


    def _validate_estimator(self):
        super()._validate_estimator()
        self.nn_m_ = check_neighbors_object(
            "m_neighbors", self.m_neighbors, additional_neighbor=1     #TODO
        )
        self.nn_m_.set_params(**{"n_jobs": self.n_jobs})


    def _fit_resample(self, X, y):
        self._validate_estimator()

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:      
                continue

            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = _safe_indexing(X, target_class_indices)

            self.nn_m_.fit(X)       
            noise,n_maj = self._in_danger_noise(
                self.nn_m_, X_class, class_sample, y, kind="noise" 
            )
            # if not any(noise):continue
            noise_index = np.flatnonzero(noise== True)      #bool
            # print(n_maj)

            def conut_weight(n_maj,noise_index):
                
                # n_maj = np.delete(n_maj,noise_index)            
                new_n_maj = [round((1-i/self.m_neighbors),2) for i in n_maj]    
                return new_n_maj
            new_n_maj = conut_weight(n_maj=n_maj,noise_index=noise_index)
            # print(new_n_maj,len(new_n_maj))


            self.nn_k_.fit(X_class)         
            nns = self.nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]      
            # distance, nns = self.nn_k_.kneighbors(X_class, return_distance=True)     
            # nns = nns[:,1:]
            # distance = distance[:,1:]
            # print(len(distance),len(nns))
            # for i in  range(len(distance)):
            #     print(distance[i],nns[i])
            # print('smote：\t',len(nns),nns)

            X_new, y_new = make_samples_zhou(
                X=X_class,
                y_dtype=y.dtype,
                y_type=class_sample, 
                nn_data=X_class, 
                nn_num=nns, 
                n_samples=n_samples,
                new_n_maj=new_n_maj,
            )
            X_resampled.append(X_new)
            y_resampled.append(y_new)

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:
            X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)


        return X_resampled, y_resampled


# @Substitution(
#     sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
#     random_state=_random_state_docstring)
class SMOTENC(SMOTE):
    """Synthetic Minority Over-sampling Technique for Nominal and Continuous
    (SMOTE-NC).

    Unlike :class:`SMOTE`, SMOTE-NC for dataset containing continuous and
    categorical features.

    Read more in the :ref:`User Guide <smote_adasyn>`.

    Parameters
    ----------
    categorical_features : ndarray of shape (n_cat_features,) or (n_features,)
        Specified which features are categorical. Can either be:

        - array of indices specifying the categorical features;
        - mask array of shape (n_features, ) and ``bool`` dtype for which
          ``True`` indicates the categorical features.

    sampling_strategy : float, str, dict or callable, default='auto'
        Sampling information to resample the data set.

        - When ``float``, it corresponds to the desired ratio of the number of
          samples in the minority class over the number of samples in the
          majority class after resampling. Therefore, the ratio is expressed as
          :math:`\\alpha_{os} = N_{rm} / N_{M}` where :math:`N_{rm}` is the
          number of samples in the minority class after resampling and
          :math:`N_{M}` is the number of samples in the majority class.

            .. warning::
               ``float`` is only available for **binary** classification. An
               error is raised for multi-class classification.

        - When ``str``, specify the class targeted by the resampling. The
          number of samples in the different classes will be equalized.
          Possible choices are:

            ``'minority'``: resample only the minority class;

            ``'not minority'``: resample all classes but the minority class;

            ``'not majority'``: resample all classes but the majority class;

            ``'all'``: resample all classes;

            ``'auto'``: equivalent to ``'not majority'``.

        - When ``dict``, the keys correspond to the targeted classes. The
          values correspond to the desired number of samples for each targeted
          class.

        - When callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples for each class.

    random_state : int, RandomState instance, default=None
        Control the randomization of the algorithm.

        - If int, ``random_state`` is the seed used by the random number
          generator;
        - If ``RandomState`` instance, random_state is the random number
          generator;
        - If ``None``, the random number generator is the ``RandomState``
          instance used by ``np.random``.

    k_neighbors : int or object, default=5
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples.  If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.

    n_jobs : int, default=None
        Number of CPU cores used during the cross-validation loop.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.

    See Also
    --------
    SMOTE : Over-sample using SMOTE.

    SVMSMOTE : Over-sample using SVM-SMOTE variant.

    BorderlineSMOTE : Over-sample using Borderline-SMOTE variant.

    ADASYN : Over-sample using ADASYN.

    KMeansSMOTE : Over-sample applying a clustering before to oversample using
        SMOTE.

    Notes
    -----
    See the original paper [1]_ for more details.

    Supports mutli-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    See
    :ref:`sphx_glr_auto_examples_over-sampling_plot_comparison_over_sampling.py`,
    and :ref:`sphx_glr_auto_examples_over-sampling_plot_illustration_generation_sample.py`.

    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE:
       synthetic minority over-sampling technique," Journal of artificial
       intelligence research, 321-357, 2002.

    Examples
    --------

    >>> from collections import Counter
    >>> from numpy.random import RandomState
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.over_sampling import SMOTENC
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape (%s, %s)' % X.shape)
    Original dataset shape (1000, 20)
    >>> print('Original dataset samples per class {}'.format(Counter(y)))
    Original dataset samples per class Counter({1: 900, 0: 100})
    >>> # simulate the 2 last columns to be categorical features
    >>> X[:, -2:] = RandomState(10).randint(0, 4, size=(1000, 2))
    >>> sm = SMOTENC(random_state=42, categorical_features=[18, 19])
    >>> X_res, y_res = sm.fit_resample(X, y)
    >>> print('Resampled dataset samples per class {}'.format(Counter(y_res)))
    Resampled dataset samples per class Counter({0: 900, 1: 900})
    """

    def __init__(
        self,
        categorical_features,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
        )
        self.categorical_features = categorical_features

    def _check_X_y(self, X, y):
        """Overwrite the checking to let pass some string for categorical
        features.
        """
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = check_X_y(X, y, accept_sparse=["csr", "csc"], dtype=None)
        return X, y, binarize_y

    def _validate_estimator(self):
        super()._validate_estimator()
        categorical_features = np.asarray(self.categorical_features)
        if categorical_features.dtype.name == "bool":
            self.categorical_features_ = np.flatnonzero(categorical_features)
        else:
            if any(
                [
                    cat not in np.arange(self.n_features_)
                    for cat in categorical_features
                ]
            ):
                raise ValueError(
                    "Some of the categorical indices are out of range. Indices"
                    " should be between 0 and {}".format(self.n_features_)
                )
            self.categorical_features_ = categorical_features
        self.continuous_features_ = np.setdiff1d(
            np.arange(self.n_features_), self.categorical_features_
        )

    def _fit_resample(self, X, y):
        self.n_features_ = X.shape[1]
        self._validate_estimator()

        # compute the median of the standard deviation of the minority class
        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)

        X_continuous = X[:, self.continuous_features_]
        X_continuous = check_array(X_continuous, accept_sparse=["csr", "csc"])
        X_minority = _safe_indexing(
            X_continuous, np.flatnonzero(y == class_minority)
        )

        if sparse.issparse(X):
            if X.format == "csr":
                _, var = csr_mean_variance_axis0(X_minority)
            else:
                _, var = csc_mean_variance_axis0(X_minority)
        else:
            var = X_minority.var(axis=0)
        self.median_std_ = np.median(np.sqrt(var))

        X_categorical = X[:, self.categorical_features_]
        if X_continuous.dtype.name != "object":
            dtype_ohe = X_continuous.dtype
        else:
            dtype_ohe = np.float64
        self.ohe_ = OneHotEncoder(
            sparse=True, handle_unknown="ignore", dtype=dtype_ohe
        )
        # the input of the OneHotEncoder needs to be dense
        X_ohe = self.ohe_.fit_transform(
            X_categorical.toarray()
            if sparse.issparse(X_categorical)
            else X_categorical
        )

        # we can replace the 1 entries of the categorical features with the
        # median of the standard deviation. It will ensure that whenever
        # distance is computed between 2 samples, the difference will be equal
        # to the median of the standard deviation as in the original paper.
        X_ohe.data = (
            np.ones_like(X_ohe.data, dtype=X_ohe.dtype) * self.median_std_ / 2
        )
        X_encoded = sparse.hstack((X_continuous, X_ohe), format="csr")

        X_resampled, y_resampled = super()._fit_resample(X_encoded, y)

        # reverse the encoding of the categorical features
        X_res_cat = X_resampled[:, self.continuous_features_.size:]
        X_res_cat.data = np.ones_like(X_res_cat.data)
        X_res_cat_dec = self.ohe_.inverse_transform(X_res_cat)

        if sparse.issparse(X):
            X_resampled = sparse.hstack(
                (
                    X_resampled[:, : self.continuous_features_.size],
                    X_res_cat_dec,
                ),
                format="csr",
            )
        else:
            X_resampled = np.hstack(
                (
                    X_resampled[:, : self.continuous_features_.size].toarray(),
                    X_res_cat_dec,
                )
            )

        indices_reordered = np.argsort(
            np.hstack((self.continuous_features_, self.categorical_features_))
        )
        if sparse.issparse(X_resampled):
            # the matrix is supposed to be in the CSR format after the stacking
            col_indices = X_resampled.indices.copy()
            for idx, col_idx in enumerate(indices_reordered):
                mask = X_resampled.indices == col_idx
                col_indices[mask] = idx
            X_resampled.indices = col_indices
        else:
            X_resampled = X_resampled[:, indices_reordered]

        return X_resampled, y_resampled

    def _generate_samples(self, X, nn_data, nn_num, rows, cols, steps):
        """Generate a synthetic sample with an additional steps for the
        categorical features.

        Each new sample is generated the same way than in SMOTE. However, the
        categorical features are mapped to the most frequent nearest neighbors
        of the majority class.
        """
        rng = check_random_state(self.random_state)
        X_new = super()._generate_samples(
            X, nn_data, nn_num, rows, cols, steps
        )
        # change in sparsity structure more efficient with LIL than CSR
        X_new = (X_new.tolil() if sparse.issparse(X_new) else X_new)

        # convert to dense array since scipy.sparse doesn't handle 3D
        nn_data = (nn_data.toarray() if sparse.issparse(nn_data) else nn_data)
        all_neighbors = nn_data[nn_num[rows]]

        categories_size = [self.continuous_features_.size] + [
            cat.size for cat in self.ohe_.categories_
        ]

        for start_idx, end_idx in zip(np.cumsum(categories_size)[:-1],
                                    np.cumsum(categories_size)[1:]):
            col_maxs = all_neighbors[:, :, start_idx:end_idx].sum(axis=1)
            # tie breaking argmax
            is_max = np.isclose(col_maxs, col_maxs.max(axis=1, keepdims=True))
            max_idxs = rng.permutation(np.argwhere(is_max))
            xs, idx_sels = np.unique(max_idxs[:, 0], return_index=True)
            col_sels = max_idxs[idx_sels, 1]

            ys = start_idx + col_sels
            X_new[:, start_idx:end_idx] = 0
            X_new[xs, ys] = 1

        return X_new


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class KMeansSMOTE(BaseSMOTE):
    """Apply a KMeans clustering before to over-sample using SMOTE.

    This is an implementation of the algorithm described in [1]_.

    Read more in the :ref:`User Guide <smote_adasyn>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    k_neighbors : int or object, default=2
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples.  If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.

    {n_jobs}

    kmeans_estimator : int or object, default=None
        A KMeans instance or the number of clusters to be used. By default,
        we used a :class:`sklearn.cluster.MiniBatchKMeans` which tend to be
        better with large number of samples.

    cluster_balance_threshold : "auto" or float, default="auto"
        The threshold at which a cluster is called balanced and where samples
        of the class selected for SMOTE will be oversampled. If "auto", this
        will be determined by the ratio for each class, or it can be set
        manually.

    density_exponent : "auto" or float, default="auto"
        This exponent is used to determine the density of a cluster. Leaving
        this to "auto" will use a feature-length based exponent.

    Attributes
    ----------
    kmeans_estimator_ : estimator
        The fitted clustering method used before to apply SMOTE.

    nn_k_ : estimator
        The fitted k-NN estimator used in SMOTE.

    cluster_balance_threshold_ : float
        The threshold used during ``fit`` for calling a cluster balanced.

    See Also
    --------
    SMOTE : Over-sample using SMOTE.

    SVMSMOTE : Over-sample using SVM-SMOTE variant.

    BorderlineSMOTE : Over-sample using Borderline-SMOTE variant.

    ADASYN : Over-sample using ADASYN.

    References
    ----------
    .. [1] Felix Last, Georgios Douzas, Fernando Bacao, "Oversampling for
    Imbalanced Learning Based on K-Means and SMOTE"
    https://arxiv.org/abs/1711.00837

    Examples
    --------

    >>> import numpy as np
    >>> from imblearn.over_sampling import KMeansSMOTE
    >>> from sklearn.datasets import make_blobs
    >>> blobs = [100, 800, 100]
    >>> X, y  = make_blobs(blobs, centers=[(-10, 0), (0,0), (10, 0)])
    >>> # Add a single 0 sample in the middle blob
    >>> X = np.concatenate([X, [[0, 0]]])
    >>> y = np.append(y, 0)
    >>> # Make this a binary classification problem
    >>> y = y == 1
    >>> sm = KMeansSMOTE(random_state=42)
    >>> X_res, y_res = sm.fit_resample(X, y)
    >>> # Find the number of new samples in the middle blob
    >>> n_res_in_middle = ((X_res[:, 0] > -5) & (X_res[:, 0] < 5)).sum()
    >>> print("Samples in the middle blob: %s" % n_res_in_middle)
    Samples in the middle blob: 801
    >>> print("Middle blob unchanged: %s" % (n_res_in_middle == blobs[1] + 1))
    Middle blob unchanged: True
    >>> print("More 0 samples: %s" % ((y_res == 0).sum() > (y == 0).sum()))
    More 0 samples: True
    """

    def __init__(
        self,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=2,              #TODO 5 KNN,  self.nn_k_
        n_jobs=None,
        kmeans_estimator=None,
        cluster_balance_threshold="auto",
        density_exponent="auto",
        m_neighbors=5,             #   TODO
        kind='kmeans',
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.kmeans_estimator = kmeans_estimator
        self.cluster_balance_threshold = cluster_balance_threshold
        self.density_exponent = density_exponent
        self.m_neighbors = m_neighbors  # TODO
        self.k_neighbors = k_neighbors
        self.kind = kind

    def _validate_estimator(self,n_clusters_zhou):
        super()._validate_estimator()           #nn_k_
        self.nn_m_ = check_neighbors_object(
            "m_neighbors", self.m_neighbors, additional_neighbor=1            
        )
        self.nn_m_.set_params(**{"n_jobs": self.n_jobs})


        if self.kmeans_estimator is None:
            self.kmeans_estimator_ = MiniBatchKMeans(
            # self.kmeans_estimator_ = KMeans(
                # n_clusters=5,          #TODO
                random_state=self.random_state,
            )
        elif isinstance(self.kmeans_estimator, int):
            print('bbb')
            self.kmeans_estimator_ = MiniBatchKMeans(
                n_clusters=self.kmeans_estimator,
                random_state=self.random_state,
            )
        else:
            self.kmeans_estimator_ = clone(self.kmeans_estimator)       

        # validate the parameters
        for param_name in ("cluster_balance_threshold", "density_exponent"):
            param = getattr(self, param_name)
            if isinstance(param, str) and param != "auto":
                raise ValueError(
                    "'{}' should be 'auto' when a string is passed. "
                    "Got {} instead.".format(param_name, repr(param))
                )
        self.cluster_balance_threshold_ = (
            self.cluster_balance_threshold
            if self.kmeans_estimator_.n_clusters != 1
            else -np.inf
        )

    def _find_cluster_sparsity(self, X):
        """Compute the cluster sparsity."""
        euclidean_distances = pairwise_distances(
            X, metric="euclidean", n_jobs=self.n_jobs
        )
        # negate diagonal elements
        for ind in range(X.shape[0]):
            euclidean_distances[ind, ind] = 0

        non_diag_elements = (X.shape[0] ** 2) - X.shape[0]
        mean_distance = euclidean_distances.sum() / non_diag_elements
        exponent = (
            math.log(X.shape[0], 1.6) ** 1.8 * 0.16
            if self.density_exponent == "auto"
            else self.density_exponent
        )
        return (mean_distance ** exponent) / X.shape[0]


    def _fit_resample(self, X, y):
        # print(':\t',len(X))
        X_resampled = X.copy()
        y_resampled = y.copy()


        if len(X_resampled)<100:n_clusters_zhou =5
        elif  len(X_resampled)<500:n_clusters_zhou = 8
        elif len(X_resampled) <1000:n_clusters_zhou = 15
        else: n_clusters_zhou = 30
        self._validate_estimator(n_clusters_zhou=n_clusters_zhou)
        total_inp_samples = sum(self.sampling_strategy_.values())


        for class_sample, n_samples in self.sampling_strategy_.items():
            

            if n_samples == 0:      
                continue

            X_clusters = self.kmeans_estimator_.fit_predict(X)      
            # print(':\t',Counter(X_clusters))
            # X_1 = pd.DataFrame(X)
            # plt.scatter(X_1[0],X_1[1],c=X_clusters,alpha=0.5)
            valid_clusters = []
            cluster_sparsities = []


            '''Filter and select clusters for sampling, select clusters with more minority classes, threshold 0.5'''
            for cluster_idx in range(self.kmeans_estimator_.n_clusters):        

                cluster_mask = np.flatnonzero(X_clusters == cluster_idx)    
                X_cluster = _safe_indexing(X, cluster_mask)     
                y_cluster = _safe_indexing(y, cluster_mask)     

                cluster_class_mean = (y_cluster == class_sample).mean()     

                if self.cluster_balance_threshold_ == "auto":       
                    balance_threshold = n_samples / total_inp_samples / 2       #TODO
                else:balance_threshold = self.cluster_balance_threshold_        

                # the cluster is already considered balanced
                if cluster_class_mean < balance_threshold:  
                    continue

                # not enough samples to apply SMOTE
                anticipated_samples = cluster_class_mean * X_cluster.shape[0]
                if anticipated_samples < self.nn_k_.n_neighbors:
                    continue

                X_cluster_class = _safe_indexing(   
                    X_cluster, np.flatnonzero(y_cluster == class_sample)
                )

                valid_clusters.append(cluster_mask)
                cluster_sparsities.append(
                    self._find_cluster_sparsity(X_cluster_class)
                )

            cluster_sparsities = np.array(cluster_sparsities)
            cluster_weights = cluster_sparsities / cluster_sparsities.sum()

            if not valid_clusters:          
                print('valid_clusters',valid_clusters,class_sample,'------------------------------------------------------------------------------')
                raise RuntimeError(
                    "No clusters found with sufficient samples of "
                    "class {}. Try lowering the cluster_balance_threshold "
                    "or increasing the number of "
                    "clusters.".format(class_sample)
                )
                # continue


            
            for valid_cluster_idx, valid_cluster in enumerate(valid_clusters):      
                X_cluster = _safe_indexing(X, valid_cluster)        
                y_cluster = _safe_indexing(y, valid_cluster)        

                X_cluster_class = _safe_indexing(       
                    X_cluster, np.flatnonzero(y_cluster == class_sample)
                )
                # print(type(X_cluster_class),len(X_cluster_class),X_cluster_class)


                
                if (n_samples * cluster_weights[valid_cluster_idx])%1 >=0.5 :
                    cluster_n_samples = int(
                        math.ceil(n_samples * cluster_weights[valid_cluster_idx])
                    )
                elif (n_samples * cluster_weights[valid_cluster_idx])%1 <0.5:
                    cluster_n_samples = int(
                        math.floor(n_samples * cluster_weights[valid_cluster_idx])
                    )


                self.nn_m_.fit(X)       
                noise, n_maj = self._in_danger_noise(
                    self.nn_m_, X_cluster_class, class_sample, y, kind="noise"
                )

                danger, n_maj = self._in_danger_noise(
                    self.nn_m_, X_cluster_class, class_sample, y, kind="danger"
                )
                

                # print('\nnoise:\t', Counter(noise))
                noise_index = np.flatnonzero(noise == True)
                # print('noise_index:\t', noise_index)
                # print('danger:\t',Counter(danger))
                danger_index = np.flatnonzero(danger == True)
                # print('danger_index:\t',danger_index)

                danger_and_safe = (noise == False)      
                # print('danger_and_safe:\n',danger_and_safe)


                self.nn_k_.fit(X_cluster_class)
                nns = self.nn_k_.kneighbors(            
                    X_cluster_class, return_distance=False
                )[:, 1:]
                # print('nns:\t',len(nns))

                
                def conut_weight(n_maj, noise_index):
                    # n_maj = np.delete(n_maj, noise_index)  #TODO 
                    new_n_maj = [round((1 - i / self.m_neighbors), 2) for i in n_maj]
                    return new_n_maj
                new_n_maj = conut_weight(n_maj=n_maj, noise_index=noise_index)
                # print('\n权重:\t',len(new_n_maj),new_n_maj)


                if self.kind == 'kmeans':
                    X_new, y_new = self._make_samples(
                        X=X_cluster_class,
                        y_dtype=y.dtype,
                        y_type=class_sample,
                        nn_data=X_cluster_class,
                        nn_num=nns,
                        n_samples=cluster_n_samples,
                        step_size=1.0,
                    )

                elif self.kind == 'kmeans-borderline' and cluster_n_samples !=0:
                    # print(cluster_n_samples)
                    X_new, y_new = make_samples_zhou(
                        X=X_cluster_class,
                        y_dtype=y.dtype,
                        y_type=class_sample,
                        nn_data=X_cluster_class,
                        nn_num=nns,
                        n_samples=cluster_n_samples,        
                        new_n_maj=new_n_maj,
                        # danger_and_safe=danger_and_safe,
                    )


            
                X_resampled = np.vstack((X_resampled,X_new))
                y_resampled = np.hstack((y_resampled, y_new))

        return X_resampled, y_resampled
