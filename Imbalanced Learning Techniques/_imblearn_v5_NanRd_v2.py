'''
NanRD-weight-version,
***Distribute the number of interpolation values ​​according to the number of natural neighbors and density weights
***Select natural neighbors and calculate interpolation positions according to weights (relative density)
Modified: smote, borderline-smote1, kmeans-smote
'''
import math
from collections import Counter
import numpy as np
from scipy import sparse
from sklearn import neighbors
from sklearn.base import clone
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from sklearn.svm import SVC
from sklearn.utils import check_random_state
from sklearn.utils import _safe_indexing
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.exceptions import raise_isinstance_error
from imblearn.utils import check_neighbors_object
from imblearn.utils import Substitution
from imblearn.utils._docstring import _n_jobs_docstring
from imblearn.utils._docstring import _random_state_docstring
import random
from sklearn.preprocessing import scale
import math


def generate_samples_Lm(X, y_type, nn_data, nn_num, n_samples,weights):
    """
        The number of interpolation values ​​for each seed sample is allocated according to the number of natural neighbors

        Parameters
        ----------
        X :                 ndarray.   seed samples
        y_type :            label of minority.
        nn_data :           ndarray.    Data set carrying all the neighbours to be used
        nn_num :            ndarray of shape (n_samples_all, k_nearest_neighbours)
                            The nearest neighbours of each sample in `nn_data`.
        n_samples :         int.    The number of samples to generate.
        weights :           ndarray. The weights.

        Returns
        -------
        X_new :     ndarray. synthetically generated samples.
        y_new :     ndarray shape (n_samples_new,). labels for synthetic samples.
    """
    
    nans_ordered = []                             
    for i in range(len(nn_num)):                    
        values = weights[nn_num[i]] 
        keys = nn_num[i]   
        dicts,num = {},[]
        for j in range(len(values)):dicts[keys[j]] = values[j]
        d_order=sorted(dicts.items(),key=lambda x:x[1],reverse=False)   
        for d in d_order:num.append(d[0])
        if num != []:nans_ordered.append(num)   
    nonzero_index = np.nonzero(nn_num) 
    X = X[nonzero_index]


    
    neighbors_num = [len(i) for i in nans_ordered]  
    seeds_index =  [0]*len(X)                                                   # len(X) = len(seeds_index) = len(neighbors_num)
    for i in range(len(seeds_index)):
        index = np.where(nn_data == X[i])
        index = list(index[0])
        seeds_index[i] = max(set(index),key=index.count)
    weights_seeds = weights[seeds_index]    
    where_are_inf = np.isinf(weights_seeds) # TODO:
    weights_seeds[where_are_inf] = 0.1
    neighbors_num_scale = scale(X=neighbors_num,with_mean=True,with_std=True,copy=True) 
    weights_scale = scale(X=weights_seeds,with_mean=True,with_std=True,copy=True)   
    w = [math.atan((abs(neighbors_num_scale[i] * weights_scale[i]))**0.5) for i in range(len(weights_scale))]
    new_num = [round(n_samples*(w[i] / sum(w)),0) for i in range(len(w))]


   # Repeatedly select sum(new_num) times the index of the base sample
    base_indices = []   # len(base_indices) == sum(new_num)
    for i ,v in enumerate(new_num):base_indices.extend([i] * int(v))


#All natural neighbor indexes, sum(new_num) times, select natural neighbors from large to small
    Nan_indices = []    # len(Nan_indices) == sum(new_num)
    for i in range(len(nans_ordered)):
# Number of rounds = number of seeds to be generated // number of natural neighbors of each seed
# The rest = number of seeds to be generated % number of natural neighbors of each seed
        rounds = int(new_num[i] // len(nans_ordered[i]))
        rest = int(new_num[i] % len(nans_ordered[i]))
        Nan_indices.extend(nans_ordered[i] * rounds)
        Nan_indices.extend(nans_ordered[i][:rest])


    
    X_base = X[base_indices]
    X_neighbor = nn_data[Nan_indices]
    X_base_weight = weights_seeds[base_indices]
    X_neighbor_weight = weights[Nan_indices]

    proportions = []
    for n in range(len(base_indices)):  # len(base_indices) 
        if X_base_weight[n]!=0 and X_neighbor_weight[n]!=0: 
            if X_base_weight[n]>= X_neighbor_weight[n]:
                proportion = (X_neighbor_weight[n] / (X_base_weight[n]+X_neighbor_weight[n])*round(random.uniform(0,1),len(str(len(base_indices)))))
            elif X_base_weight[n]< X_neighbor_weight[n]:
                proportion = X_neighbor_weight[n] / (X_base_weight[n]+X_neighbor_weight[n])
                proportion = proportion+(1-proportion)*(round(random.uniform(0,1),len(str(len(base_indices)))))
        proportions.append(proportion)

    proportions=np.array(proportions).reshape(int(len(proportions)),1)
    samples= X_base + np.multiply(proportions, X_neighbor - X_base)
    return samples,np.array([y_type]*len(samples))


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
            "k_neighbors", self.k_neighbors, additional_neighbor=1
        )
        self.nn_k_.set_params(**{"n_jobs": self.n_jobs})


    def _in_danger_noise(
        self, nn_estimator, samples, target_class, y, kind="danger"
    ):
        """Estimate if a set of sample are in danger or noise.
        Parameters
        ----------
        nn_estimator :  estimator
                        An estimator that inherits from
                        :class:`sklearn.neighbors.base.KNeighborsMixin` use to determine if
                        a sample is in danger/noise.
        samples :       {array-like, sparse matrix} of shape (n_samples, n_features)
                        The samples to check if either they are in danger or not.
        target_class :  int or str
                        The target corresponding class being over-sampled.
        y :             array-like of shape (n_samples,)
                        The true label in order to check the neighbour labels.
        kind :          {'danger', 'noise'}, default='danger'
                        The type of classification to use. Can be either:
                        - If 'danger', check if samples are in danger,
                        - If 'noise', check if samples are noise.
        Returns
        -------
        output :    ndarray of shape (n_samples,)
                    A boolean array where True refer to samples in danger or noise.
        """
        x = nn_estimator.kneighbors(samples, return_distance=False)[:, 1:]
        nn_label = (y[x] != target_class).astype(int)
        n_maj = np.sum(nn_label, axis=1)

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
        k_neighbors=5,      #nn_k, 
        n_jobs=None,
        m_neighbors=10,     #nn_m
        weight = None,      
        nans = None,        
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.m_neighbors = m_neighbors
        self.k_neighbors = k_neighbors
        self.weight = weight
        self.nans = nans

    def _validate_estimator(self):
        super()._validate_estimator()
        self.nn_m_ = check_neighbors_object(
            "m_neighbors", self.m_neighbors, additional_neighbor=1    
        )
        self.nn_m_.set_params(**{"n_jobs": self.n_jobs})


    def _fit_resample(self, X, y):
        self._validate_estimator()
        X_resampled = X.copy()  
        y_resampled = y.copy()

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:continue

            target_class_indices:np.ndarray = np.flatnonzero(self.weight > 0)          
            X_class:np.ndarray = _safe_indexing(X, target_class_indices)               
            weight_min:np.ndarray = _safe_indexing(self.weight, target_class_indices)  
            nans:np.ndarray = _safe_indexing(self.nans, target_class_indices)          

            
            self.nn_m_.fit(X[np.flatnonzero(self.weight >= 0)])     # KNN
            danger_index ,n_maj = self._in_danger_noise(
                self.nn_m_, X_class, class_sample, y, kind="danger")
            if not any(danger_index):continue
            seed_nns = nans[danger_index]                   

            nans_new = [0]*len(seed_nns)
            for i in range(len(seed_nns)):
                nns = []
                for nn in seed_nns[i]:
                    index = np.where(target_class_indices == nn)
                    if len(index[0]) != 0: 
                        nns.append(index[0][0])
                nans_new[i] = nns

            X_new, y_new = generate_samples_Lm(
                X=_safe_indexing(X_class, danger_index),    # seeds:
                y_type=class_sample,                        
                nn_data=X_class,                                            
                nn_num=nans_new,                            # seeds
                n_samples=n_samples,                                          
                weights=weight_min,                         
            )

            
            if sparse.issparse(X_new): X_resampled = sparse.vstack([X_resampled, X_new])
            else:X_resampled = np.vstack((X_resampled, X_new))    
            y_resampled = np.hstack((y_resampled, y_new))
            return  X_resampled,y_resampled


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class SVMSMOTE(BaseSMOTE):
    """
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
        weight:np.array = None,      
        nans:np.array = None,        
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
        self.weight=weight
        self.nans=nans

    def _validate_estimator(self):
        super()._validate_estimator()
        self.nn_m_ = check_neighbors_object(
            "m_neighbors",self.m_neighbors, additional_neighbor=1     
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
            if n_samples == 0: continue

            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = _safe_indexing(X, target_class_indices)   
            weight_maj = _safe_indexing(self.weight,target_class_indices)       
            new_n_maj = [round((1-i/self.ntree),2) for i in weight_maj] 

            
            self.svm_estimator_.fit(X, y)
            support_index = self.svm_estimator_.support_[           # np.ndarray
                y[self.svm_estimator_.support_] == class_sample]    
            support_vector = _safe_indexing(X, support_index)       

            
            self.nn_m_.fit(X)           
            noise_bool = self._in_danger_noise(
                self.nn_m_, support_vector, class_sample, y, kind="noise")[0] 
            noise_index = np.flatnonzero(noise_bool)
            support_index = np.delete(support_index,noise_index)    

            
            support_vector = _safe_indexing(
                support_vector, np.flatnonzero(np.logical_not(noise_bool)))       

            
            danger_bool = self._in_danger_noise(
                self.nn_m_, support_vector, class_sample, y, kind="danger")[0]      
            safety_bool = np.logical_not(danger_bool)       
            danger_index = np.delete(support_index,np.flatnonzero(safety_bool))
            safe_index = np.delete(support_index,np.flatnonzero(danger_bool))

            
            danger_list,safe_list = [],[]
            for i in danger_index:
                ii = np.where(target_class_indices == i)
                danger_list.append(ii[0][0])
            for i in safe_index:
                ii = np.where(target_class_indices == i)
                safe_list.append(ii[0][0])

            self.nn_k_.fit(X_class)     
            fractions = random_state.beta(10, 10)
            n_generated_samples = int(fractions * (n_samples + 1))

            
            if np.count_nonzero(danger_bool) > 0:
                nns = self.nn_k_.kneighbors(
                    _safe_indexing(support_vector, np.flatnonzero(danger_bool)),
                    return_distance=False,)[:, 1:]  
                X_new_1, y_new_1 = generate_samples_zhou(
                    X=_safe_indexing(support_vector, np.flatnonzero(danger_bool)),
                    y_dtype=y.dtype,
                    y_type=class_sample,
                    nn_data=X_class,                
                    nn_num=nns,                     
                    n_samples=n_generated_samples,      
                    weights=new_n_maj,            
                    danger_and_safe=len(danger_index),      
                    mother_point = np.array(danger_list),   
                )

            
            if np.count_nonzero(safety_bool) > 0:
                nns = self.nn_k_.kneighbors(
                    _safe_indexing(support_vector, np.flatnonzero(safety_bool)),
                    return_distance=False,)[:, 1:]
                X_new_2, y_new_2 = generate_samples_zhou(
                    X=_safe_indexing(support_vector, np.flatnonzero(safety_bool)),
                    y_dtype=y.dtype,
                    y_type=class_sample,
                    nn_data=X_class,
                    nn_num=nns,
                    n_samples=n_samples - n_generated_samples,
                    weights=new_n_maj,
                    danger_and_safe=len(safe_index),
                    mother_point=np.array(safe_list),
                )

            if (
                np.count_nonzero(danger_bool) > 0
                and np.count_nonzero(safety_bool) > 0
            ):
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
    """

    def __init__(
        self,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
        weight:np.array = None,      
        nans:np.array = None,        
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.weight=weight
        self.nans=nans

    def _fit_resample(self, X, y):
        self._validate_estimator()

        X_resampled = [X.copy()]    
        y_resampled = [y.copy()]

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:continue    

            target_class_indices = np.flatnonzero(self.weight >0)           
            X_class = _safe_indexing(X, target_class_indices)               
            weight_min = _safe_indexing(self.weight, target_class_indices)  
            nans = _safe_indexing(self.nans, target_class_indices)          

            
            nans_new = [0]*len(X_class)
            for i in range(len(nans)):
                nns = []
                for nn in nans[i]:
                    index = np.where(target_class_indices == nn)
                    if len(index[0]) != 0: 
                        nns.append(index[0][0])
                nans_new[i] = nns

            
            X_new, y_new = generate_samples_Lm(
                X=X_class,              # seeds: 
                y_type=class_sample,    
                nn_data=X_class,        
                nn_num=nans_new,        
                n_samples=n_samples,    
                weights=weight_min,     
             )

            X_resampled.append(X_new)
            y_resampled.append(y_new)

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)
        return X_resampled, y_resampled


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class KMeansSMOTE(BaseSMOTE):
    """
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
        k_neighbors=2,              
        n_jobs=None,
        kmeans_estimator=None,
        cluster_balance_threshold="auto",
        density_exponent="auto",
        weight:np.array = None,      
        nans:np.array = None,        
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
        self.k_neighbors = k_neighbors
        self.weight=weight
        self.nans=nans

    def _validate_estimator(self,n_clusters_zhou=30):
        super()._validate_estimator()           #nn_k_
        if self.kmeans_estimator is None:
            self.kmeans_estimator_ = MiniBatchKMeans(
                n_clusters=n_clusters_zhou,          #TODO:
                random_state=self.random_state,
            )
        elif isinstance(self.kmeans_estimator, int):
            self.kmeans_estimator_ = MiniBatchKMeans(
                n_clusters=self.kmeans_estimator,
                random_state=self.random_state,)
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
            X, metric="euclidean", n_jobs=self.n_jobs)
        # negate diagonal elements
        for ind in range(X.shape[0]):euclidean_distances[ind, ind] = 0

        non_diag_elements = (X.shape[0] ** 2) - X.shape[0]
        mean_distance = euclidean_distances.sum() / non_diag_elements
        exponent = (
            math.log(X.shape[0], 1.6) ** 1.8 * 0.16
            if self.density_exponent == "auto"
            else self.density_exponent)
        return (mean_distance ** exponent) / X.shape[0]


    def _fit_resample(self, X, y):
        X_resampled = X.copy()
        y_resampled = y.copy()

        if len(X_resampled)<100:n_clusters_zhou =5
        elif  len(X_resampled)<500:n_clusters_zhou = 8
        elif len(X_resampled) <1000:n_clusters_zhou = 15
        else: n_clusters_zhou = 50

        self._validate_estimator(n_clusters_zhou=n_clusters_zhou)
        total_inp_samples = sum(self.sampling_strategy_.values())
        # print(':\t',self.kmeans_estimator_.n_clusters)
        
        for class_sample, n_samples in self.sampling_strategy_.items():
            '''Step_3: Oversample each selected cluster'''
            if n_samples == 0:continue      
            X_clusters = self.kmeans_estimator_.fit_predict(X)  
            valid_clusters = []             
            cluster_sparsities = []

            # print(':\n',X_clusters,len(X_clusters))
            '''Step_2: Filter clusters for sampling, select clusters with more minority classes, threshold 0.5'''
            for cluster_idx in range(self.kmeans_estimator_.n_clusters):        
                cluster_mask = np.flatnonzero(X_clusters == cluster_idx)    
                X_cluster = _safe_indexing(X, cluster_mask)     
                y_cluster = _safe_indexing(y, cluster_mask)     
                cluster_class_mean = (y_cluster == class_sample).mean()     
                # print(cluster_idx,cluster_mask)

                if self.cluster_balance_threshold_ == "auto":       # TODO
                    balance_threshold = n_samples / total_inp_samples / 2       
                    balance_threshold = 0.1
                else:balance_threshold = self.cluster_balance_threshold_        

                # the cluster is already considered balanced
                if cluster_class_mean < balance_threshold:continue 

                # not enough samples to apply SMOTE
                anticipated_samples = cluster_class_mean * X_cluster.shape[0]
                if anticipated_samples < self.nn_k_.n_neighbors:continue

                X_cluster_class = _safe_indexing(   
                    X_cluster, np.flatnonzero(y_cluster == class_sample))

                valid_clusters.append(cluster_mask)
                cluster_sparsities.append(
                    self._find_cluster_sparsity(X_cluster_class))

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

            '''Step_3: Oversample each selected cluster'''
            for valid_cluster_idx, valid_cluster in enumerate(valid_clusters):      
                X_cluster = _safe_indexing(X, valid_cluster)                
                y_cluster = _safe_indexing(y, valid_cluster)                
                weight_cluster = _safe_indexing(self.weight,valid_cluster)      
                target_class_index = np.flatnonzero(weight_cluster > 0)     
                weight_min = weight_cluster[target_class_index]     


                
                X_cluster_class = X_cluster[target_class_index]  

                
                cluster_nns = self.nans[valid_cluster]
                seed_nns = cluster_nns[target_class_index]

                
                nans_new = [[]]*len(seed_nns)
                seed_index = []
                for i in range(len(seed_nns)):
                    nns = []
                    for nn in seed_nns[i]:
                        index = np.where(valid_cluster == nn)
                        if len(index[0]) != 0: 
                            nns.append(index[0][0])
                    if nns != []:seed_index.append(i)
                    nans_new[i] = nns

                seed_index = np.array(seed_index)
                if len(seed_index) == 0:
                    continue
                nans_new = np.array(nans_new)[seed_index]       
                weight_min = weight_min[seed_index]
                X_cluster_class = X_cluster_class[seed_index]

                
                target_class_index = target_class_index[seed_index]
                nans_new_2 = [[]]*len(seed_index)
                for i in range(len(nans_new)):
                    nns = []
                    for nn in nans_new[i]:
                        index = np.where(target_class_index == nn)
                        if len(index[0]) != 0: 
                            nns.append(index[0][0])
                    nans_new_2[i] = nns

                # cluster_n_samples
                if (n_samples * cluster_weights[valid_cluster_idx])%1 >=0.5 :
                    cluster_n_samples = int(math.ceil(n_samples * cluster_weights[valid_cluster_idx]))
                elif (n_samples * cluster_weights[valid_cluster_idx])%1 <0.5:
                    cluster_n_samples = int(math.floor(n_samples * cluster_weights[valid_cluster_idx]))

                
                if cluster_n_samples !=0:
                    X_new, y_new = generate_samples_Lm(
                        X=X_cluster_class,                  # seeds
                        y_type=class_sample,
                        nn_data=X_cluster_class,            
                        nn_num=nans_new_2,
                        n_samples=cluster_n_samples,        
                        weights=weight_min,
                      )

                    X_resampled = np.vstack((X_resampled,X_new))
                    y_resampled = np.hstack((y_resampled, y_new))
        return X_resampled, y_resampled
