''''crf-weight-version
smote,borderline-smote1,svmsmote,kmeans-smote
'''
import math
from collections import Counter
import numpy as np
from scipy import sparse
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


#-----------------------------------------------------------------------------------
def make_samples_zhou(
        X, y_dtype, y_type, nn_data, nn_num, n_samples,
        new_n_maj, danger_and_safe=None, mother_point=None
):
    """
        Parameters
        ----------
        X :         {array-like, sparse matrix} of shape (n_samples, n_features)
                    Points from which the points will be created.
        y_dtype :   data type，The data type of the targets.
        y_type :    str or int
                    The minority target value, just so the function can return the
                    target values for the synthetic variables with correct length in
                    a clear format.
        nn_data :   ndarray of shape (n_samples_all, n_features)
                    Data set carrying all the neighbours to be used
        nn_num :    ndarray of shape (n_samples_all, k_nearest_neighbours)
                    The nearest neighbours of each sample in `nn_data`.
        n_samples : int
                    The number of samples to generate.
        step_size : float, default=1.0
                    The step size to create samples.

        Returns
        -------
        X_new :     {ndarray, sparse matrix} of shape (n_samples_new, n_features)
                    synthetically generated samples.
        y_new :     ndarray of shape (n_samples_new,)
                    Target values for synthetic samples.
    """
    X_new = generate_samples_zhou(X, nn_data, nn_num, n_samples,new_n_maj, danger_and_safe,mother_point)
    y_new = np.full(len(X_new), fill_value=y_type, dtype=y_dtype)
    return X_new, y_new


def generate_samples_zhou(X, nn_data, nn_num, n_samples,
                        new_n_maj, danger_and_safe,mother_point=None):
    '''
#Traverse the boundary and safe points, find the corresponding KNN points according to the neighbor point index matrix, randomly extract with replacement N times, subtract the coordinates of the two points, and interpolate the distance according to the weight ratio.
#The remaining m points are used to randomly extract points without replacement, calculate the weight ratio of the extracted points and the neighboring points, and each point is interpolated only once
    '''
    mother_points = nn_data[mother_point]   
    n = n_samples // danger_and_safe        
    m = n_samples % danger_and_safe         
    weidu = X.shape[1]
    X_new_1 = np.zeros(weidu)
    if not isinstance(nn_data, list):nn_data = nn_data.tolist()

    '''Sort the entire neighbor point index matrix by weight from small to large'''
    nn_num_ordered = []
    for i in range(len(nn_num)):
        values = np.array(new_n_maj)[nn_num[i]]
        keys = nn_num[i]
        dicts,num = {},[]
        for j in range(len(values)):dicts[keys[j]] = values[j]
        d_order=sorted(dicts.items(),key=lambda x:x[1],reverse=False)   
        for d in d_order:num.append(d[0])
        nn_num_ordered.append(num)
    length_of_num = len(num)        
    # if not isinstance(nn_data, list):nn_data = nn_data.tolist()

    '''N number of points to be interpolated at each point'''
    for nn in range(danger_and_safe):  
        #step_1: 
        num = nn_num_ordered[nn]    
        nn_point = mother_points[nn]      
        nn_weight = new_n_maj[mother_point[nn]]      

        for i in range(n):  
            #step_2:    
            random_point = num[i%length_of_num]           
            random_point_weight = new_n_maj[random_point]  
            random_point_data = nn_data[random_point]  

            #step_3:    
            if nn_weight != 0 and random_point_weight != 0:  
                proportion = (random_point_weight / (nn_weight + random_point_weight))  
                if nn_weight >= random_point_weight:
                    X_new_zhou = np.array(nn_point) + (
                                np.array(random_point_data) - np.array(nn_point)) * proportion * round(
                        random.uniform(0, 1), len(str(n_samples)))
                elif nn_weight < random_point_weight:
                    X_new_zhou = np.array(random_point_data) + (np.array(nn_point) - np.array(random_point_data)) * (
                                1 - proportion) * round(random.uniform(0, 1), len(str(n_samples)))
                X_new_1 = np.vstack((X_new_zhou, X_new_1))

    
    for mm in range(m):
        #step_1:
        nn_point_index = random.choice(mother_point)        
        nn_point_weight = new_n_maj[nn_point_index]         
        nn_point = nn_data[nn_point_index]                  
        a = np.where(mother_point == nn_point_index)[0][0]  

        #step_2:
        num = nn_num[a].tolist()       
        random_point = num[0]  
        random_point_weight = new_n_maj[random_point]  
        random_point_data = nn_data[random_point]

        #step_3:
        if nn_point_weight != 0 and random_point_weight != 0:  
            proportion = (random_point_weight / (nn_point_weight + random_point_weight))  
            if nn_point_weight >= random_point_weight:
                X_new_zhou = np.array(nn_point) + (np.array(random_point_data) - np.array(nn_point)) * proportion * round(random.uniform(0, 1),len(str(n_samples)))
            elif nn_point_weight < random_point_weight:
                X_new_zhou = np.array(random_point_data) + (np.array(nn_point) - np.array(random_point_data)) * (1 - proportion) * round(random.uniform(0, 1), len(str(n_samples)))
            X_new_1 = np.vstack((X_new_zhou, X_new_1))

    X_new_1 = np.delete(X_new_1, -1, 0)
    return X_new_1.astype(X.dtype)


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
        Over-sampling using Borderline SMOTE.
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
        ntree = 10,     
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
        self.ntree = ntree

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

            target_class_indices = np.flatnonzero(y == class_sample)    
            X_class = _safe_indexing(X, target_class_indices)       
            weight_maj = _safe_indexing(self.weight,target_class_indices)       
            new_n_maj = [round((1-i/self.ntree),2) for i in weight_maj] 

            self.nn_m_.fit(X)           
            danger_index ,n_maj= self._in_danger_noise(
                self.nn_m_, X_class, class_sample, y, kind="danger")
            if not any(danger_index):continue
        
            self.nn_k_.fit(X_class)     
            nns = self.nn_k_.kneighbors(    
                _safe_indexing(X_class, danger_index), return_distance=False)[:, 1:]    

            X_new, y_new = make_samples_zhou(
                X=_safe_indexing(X_class, danger_index),          
                y_dtype=y.dtype,            
                y_type=class_sample,       
                nn_data=X_class,                                
                nn_num=nns,                
                n_samples=n_samples,                             
                new_n_maj=new_n_maj,        
                danger_and_safe=np.count_nonzero(danger_index),   
                mother_point=np.flatnonzero(danger_index),      
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
    """Over-sampling using SVM-SMOTE.
    Variant of SMOTE algorithm which use an SVM algorithm to detect sample to
    use for generating new synthetic samples as proposed in [2]_.
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
        weight = None,
        ntree=10,
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
        self.weight = weight        
        self.ntree=ntree

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

            #Find the support vector in the minority class
            self.svm_estimator_.fit(X, y)
            support_index = self.svm_estimator_.support_[           #np.ndarray
                y[self.svm_estimator_.support_] == class_sample]    
            support_vector = _safe_indexing(X, support_index)       

            #Find the index of the noise point in the support vector of the minority class, and then delete the noise point index
            self.nn_m_.fit(X)           
            noise_bool = self._in_danger_noise(
                self.nn_m_, support_vector, class_sample, y, kind="noise")[0] 
            noise_index = np.flatnonzero(noise_bool)
            support_index = np.delete(support_index,noise_index)    

            #Take out the noise points in the support vector, leaving only the feature information of the safe points and boundary points
            support_vector = _safe_indexing(
                support_vector, np.flatnonzero(np.logical_not(noise_bool)))       
            
            #Find the index of the boundary points and safe points in the support vector points in the total data set
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
                X_new_1, y_new_1 = make_samples_zhou(
                    X=_safe_indexing(support_vector, np.flatnonzero(danger_bool)),
                    y_dtype=y.dtype,
                    y_type=class_sample,
                    nn_data=X_class,                
                    nn_num=nns,                     
                    n_samples=n_generated_samples,      
                    new_n_maj=new_n_maj,            
                    danger_and_safe=len(danger_index),      
                    mother_point = np.array(danger_list),   
                )

            #Points generated based on safe points
            if np.count_nonzero(safety_bool) > 0:
                nns = self.nn_k_.kneighbors(
                    _safe_indexing(support_vector, np.flatnonzero(safety_bool)),
                    return_distance=False,)[:, 1:]
                X_new_2, y_new_2 = make_samples_zhou(
                    X=_safe_indexing(support_vector, np.flatnonzero(safety_bool)),
                    y_dtype=y.dtype,
                    y_type=class_sample,
                    nn_data=X_class,
                    nn_num=nns,
                    n_samples=n_samples - n_generated_samples,
                    new_n_maj=new_n_maj,
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
    """Class to perform over-sampling using SMOTE.
    This object is an implementation of SMOTE - Synthetic Minority
    Over-sampling Technique as presented in [1]_.
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
    """

    def __init__(
        self,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
        weight = None,
        ntree = 10,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.weight=weight
        self.ntree=ntree

    def _fit_resample(self, X, y):
        self._validate_estimator()

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:continue    

            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = _safe_indexing(X, target_class_indices)   
            weight_maj = _safe_indexing(self.weight,target_class_indices)       
            new_n_maj = [round((1-i/self.ntree),2) for i in weight_maj] 

            self.nn_k_.fit(X_class)     
            nns = self.nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]

            X_new, y_new = make_samples_zhou(
                X=X_class,
                y_dtype=y.dtype,
                y_type=class_sample, 
                nn_data=X_class, 
                nn_num=nns, 
                n_samples=n_samples,
                new_n_maj=new_n_maj,
                danger_and_safe=len(new_n_maj), 
                mother_point=np.arange(len(target_class_indices)),      
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
        weight=None,
        ntree=10,
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
        self.ntree=ntree

    def _validate_estimator(self,n_clusters_zhou):
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
            '''Step_1: Clustering'''
            if n_samples == 0:continue  
            X_clusters = self.kmeans_estimator_.fit_predict(X)  
            valid_clusters = []     
            cluster_sparsities = []

            '''Step_2: Filter and select clusters for sampling, select clusters with more minority classes, threshold 0.5'''
            for cluster_idx in range(self.kmeans_estimator_.n_clusters):        
                cluster_mask = np.flatnonzero(X_clusters == cluster_idx)    
                X_cluster = _safe_indexing(X, cluster_mask)     
                y_cluster = _safe_indexing(y, cluster_mask)     
                cluster_class_mean = (y_cluster == class_sample).mean()     

                if self.cluster_balance_threshold_ == "auto":       
                    # balance_threshold = n_samples / total_inp_samples / 2       #TODO
                    balance_threshold = 0.2
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

            '''Step_3: Oversample each selected cluster'''
            for valid_cluster_idx, valid_cluster in enumerate(valid_clusters):      
                X_cluster = _safe_indexing(X, valid_cluster)        
                y_cluster = _safe_indexing(y, valid_cluster)        
                weight_maj = _safe_indexing(self.weight,valid_cluster) 
                new_n_maj = [round((1-i/self.ntree),2) for i in weight_maj] 

                X_cluster_class = _safe_indexing(       
                    X_cluster, np.flatnonzero(y_cluster == class_sample))
                new_n_maj = _safe_indexing(new_n_maj,np.flatnonzero(y_cluster == class_sample))

                #cluster_n_samples
                if (n_samples * cluster_weights[valid_cluster_idx])%1 >=0.5 :
                    cluster_n_samples = int(
                        math.ceil(n_samples * cluster_weights[valid_cluster_idx]))
                elif (n_samples * cluster_weights[valid_cluster_idx])%1 <0.5:
                    cluster_n_samples = int(
                        math.floor(n_samples * cluster_weights[valid_cluster_idx]))

                self.nn_k_.fit(X_cluster_class)     
                nns = self.nn_k_.kneighbors(            
                    X_cluster_class, return_distance=False)[:, 1:]

                if cluster_n_samples !=0:
                    X_new, y_new = make_samples_zhou(
                        X=X_cluster_class,
                        y_dtype=y.dtype,
                        y_type=class_sample,
                        nn_data=X_cluster_class,
                        nn_num=nns,
                        n_samples=cluster_n_samples,        
                        new_n_maj=new_n_maj,
                        danger_and_safe=len(new_n_maj),     
                        mother_point=np.arange(len(new_n_maj)), 
                    )

                    X_resampled = np.vstack((X_resampled,X_new))
                    y_resampled = np.hstack((y_resampled, y_new))
        return X_resampled, y_resampled
