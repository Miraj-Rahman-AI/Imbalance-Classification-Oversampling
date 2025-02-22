import math
from collections import Counter
import  matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.cluster import MiniBatchKMeans,KMeans,DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.utils import check_random_state
from sklearn.utils import _safe_indexing
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.exceptions import raise_isinstance_error
from imblearn.utils import check_neighbors_object
from imblearn.utils import check_target_type
from imblearn.utils import Substitution
from imblearn.utils._docstring import _n_jobs_docstring
from imblearn.utils._docstring import _random_state_docstring
import random
import  pandas as pd


#-----------------------------------------------------------------------------------
def make_samples_zhou(
    X, y_dtype, y_type, nn_data, nn_num, n_samples, 
                new_n_maj,danger_and_safe=None,mother_point=None
):

    X_new = generate_samples_zhou(X, nn_data, nn_num,n_samples,
            new_n_maj,danger_and_safe)
    y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
    return X_new, y_new


def generate_samples_zhou( X, nn_data, nn_num,n_samples,
                new_n_maj,danger_and_safe):

    '''
# n = (majority class - minority class) //minority class takes integer number of points to be interpolated
# m = (majority class - minority class)%minority class takes remainder and the rest is used for random
#Traverse the boundary and safe points, find the corresponding KNN points according to the neighbor point index matrix, randomly extract with replacement N times, subtract the coordinates of the two points, calculate the distance according to the weight ratio to interpolate,
#The remaining m points are used to randomly extract points without replacement, calculate the weight ratio of the extracted points and the neighbor points, and each point is only interpolated once
    '''
    # print('danger_and_safe：\t',danger_and_safe)
    if danger_and_safe is not None:     
        nn_data = _safe_indexing(nn_data, danger_and_safe)  
    else:pass           

    # print('nn_data:\t',len(nn_data),'\n',nn_data)
    n = n_samples // len(nn_data)
    m = n_samples %  len(nn_data)
    # print('n,m,n_samples',n,m,n_samples)
    weidu = X.shape[1]
    # print(':\t',weidu)
    X_new_1 = np.zeros(weidu)
    # print(X_new_1.shape,'\n',X_new_1)

    for nn in range(len(nn_data)):      
        if nn==1:
            # break
            pass

        num = nn_num[nn].tolist()            
        if  isinstance(nn_data,list): pass 
        else:nn_data = nn_data.tolist()

        nn_point = nn_data[nn]      
        nn_weight = new_n_maj[nn]       
        # print(nn_point)
        # print(num)
        for i in range(n):            
            random_point = random.choice(num)               
            # print('randow_point_1:\t',random_point)
            # random_point_index = num.index(random_point)        
            random_point_weight = new_n_maj[random_point]           
            random_point_data = nn_data[random_point]               
            # print(random_point, num.index(random_point),random_point_data)

            if (nn_weight+random_point_weight)!=0:
                proportion = (random_point_weight/(nn_weight+random_point_weight))
            else:       
                proportion = 0.5
            
            X_new_zhou = np.array(nn_point) + (np.array(random_point_data) - np.array(nn_point))*proportion
            X_new_1 = np.vstack((X_new_zhou,X_new_1))


    
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
        if (nn_point_weight+random_point_weight)!=0:
                proportion = (random_point_weight/(nn_point_weight+random_point_weight))
        elif (nn_weight+random_point_weight)==0:       
            proportion = 0.5

        X_new_zhou = np.array(nn_point) + (np.array(random_point_data)-np.array(nn_point)) * proportion
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
        k_neighbors=5,
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
            "k_neighbors", self.k_neighbors, additional_neighbor=1
        )
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
            low=0, high=nn_num.size, size=n_samples)

        # np.newaxis for backwards compatability with random_state
        steps = step_size * random_state.uniform(size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])
        X_new = self._generate_samples(X, nn_data, nn_num, rows, cols, steps)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new


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
        # print(':\t',n_maj,len(n_maj))
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
class KMeansSMOTE(BaseSMOTE):
    """
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
    """

    def __init__(
        self,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,              # KNN,  slef.nn_k_
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
        self.m_neighbors = 10  # TODO
        self.k_neighbors = k_neighbors
        self.kind = kind

    def _validate_estimator(self,n_clusters_zhou):
        super()._validate_estimator()           
        self.nn_m_ = check_neighbors_object(
            "m_neighbors", self.m_neighbors, additional_neighbor=1            
        )
        self.nn_m_.set_params(**{"n_jobs": self.n_jobs})


        if self.kmeans_estimator is None:
            self.kmeans_estimator_ = MiniBatchKMeans(
            # self.kmeans_estimator_ = KMeans(
                # n_clusters=20,          #TODO
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

        """聚类"""
        X_clusters = self.kmeans_estimator_.fit_predict(X)      


        for class_sample, n_samples in self.sampling_strategy_.items():
            print('kemans:\t',class_sample,'\:\t',n_samples)
            if n_samples == 0:  continue    #
            # X_clusters = self.kmeans_estimator_.fit_predict(X)      
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
                    balance_threshold = n_samples / total_inp_samples / 2
                else:balance_threshold = self.cluster_balance_threshold_        

                # the cluster is already considered balanced
                if cluster_class_mean < balance_threshold:  
                    continue

                # not enough samples to apply SMOTE
                anticipated_samples = cluster_class_mean * X_cluster.shape[0]
                if anticipated_samples < self.nn_k_.n_neighbors:
                    continue

                X_cluster_class = _safe_indexing(   
                    X_cluster, np.flatnonzero(y_cluster == class_sample))

                valid_clusters.append(cluster_mask)
                cluster_sparsities.append(
                    self._find_cluster_sparsity(X_cluster_class))

            cluster_sparsities = np.array(cluster_sparsities)
            cluster_weights = cluster_sparsities / cluster_sparsities.sum()

            if not valid_clusters:          
                print('No valid_clusters',valid_clusters,class_sample,'------------------------------------------------------------------------------')
                continue
                # raise RuntimeError(
                #     "No clusters found with sufficient samples of "
                #     "class {}. Try lowering the cluster_balance_threshold "
                #     "or increasing the number of "
                #     "clusters.".format(class_sample)
                # )


            '''Oversample each filtered cluster'''
            for valid_cluster_idx, valid_cluster in enumerate(valid_clusters):      
                X_cluster = _safe_indexing(X, valid_cluster)        
                y_cluster = _safe_indexing(y, valid_cluster)        

                X_cluster_class = _safe_indexing(       
                    X_cluster, np.flatnonzero(y_cluster == class_sample))
                # print(type(X_cluster_class),len(X_cluster_class),X_cluster_class)


                
                if (n_samples * cluster_weights[valid_cluster_idx])%1 >=0.5 :
                    cluster_n_samples = int(
                        math.ceil(n_samples * cluster_weights[valid_cluster_idx]))
                elif (n_samples * cluster_weights[valid_cluster_idx])%1 <0.5:
                    cluster_n_samples = int(
                        math.floor(n_samples * cluster_weights[valid_cluster_idx]))

                self.nn_m_.fit(X)       
                noise, n_maj = self._in_danger_noise(
                    self.nn_m_, X_cluster_class, class_sample, y, kind="noise")
                danger, n_maj = self._in_danger_noise(
                    self.nn_m_, X_cluster_class, class_sample, y, kind="danger")
                print(':\t',len(n_maj))

                noise_index = np.flatnonzero(noise == True)
                # print('noise_index:\t', noise_index)
                # print('danger:\t',Counter(danger))
                danger_index = np.flatnonzero(danger == True)
                # print('danger_index:\t',danger_index)

                danger_and_safe = (noise == False)      
                # print('danger_and_safe:\n',danger_and_safe)
                # print('danger_and_safe:',len(danger_and_safe))



                
                self.nn_k_.fit(_safe_indexing(X_cluster_class,danger_and_safe))
                nns = self.nn_k_.kneighbors(            
                    _safe_indexing(X_cluster_class,danger_and_safe), return_distance=False
                )[:, 1:]
                print('nns:\t',len(nns))


                #kmeans+weight
                # self.nn_k_.fit(X_cluster_class)
                # nns = self.nn_k_.kneighbors(            
                #     X_cluster_class, return_distance=False)[:, 1:]
                # print('nns:\t',len(nns))


                
                def conut_weight(n_maj:np.ndarray, noise_index:np.ndarray):
                    n_maj = np.delete(n_maj, noise_index)  #TODO 
                    new_n_maj = [round((1 - i / self.k_neighbors), 2) for i in n_maj]
                    return new_n_maj
                new_n_maj = conut_weight(n_maj=n_maj, noise_index=noise_index)
                print('\n权重:\t',len(new_n_maj),type(new_n_maj))


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
                    X_new, y_new = make_samples_zhou(
                        X=X_cluster_class,
                        y_dtype=y.dtype,
                        y_type=class_sample,
                        nn_data=X_cluster_class,
                        nn_num=nns,
                        n_samples=cluster_n_samples,        
                        new_n_maj=new_n_maj,
                        danger_and_safe=danger_and_safe,
                    )
                else:raise KeyError("kind error")
                X_resampled = np.vstack((X_resampled,X_new))
                y_resampled = np.hstack((y_resampled, y_new))
        return X_resampled, y_resampled