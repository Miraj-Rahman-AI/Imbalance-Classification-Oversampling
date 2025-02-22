"""knn + weight(rd + number)"""
from sklearn import neighbors
from sklearn.neighbors import KDTree,NearestNeighbors
import numpy as np
from collections import Counter
from sklearn.preprocessing import scale


class Natural_Neighbor(object):

    def __init__(self):
        self.target:np.array = []           # Set of classes
        self.data:np.array = []             # Instance set
        self.relative_cox = []


    def nn_k(self,min_i,maj_i,k:int):

"""Minority class = {
Outlier density -2: no natural neighbors,
Noise point -3: ratio exceeds 0.7,
Safe point: natural neighbors are all minority class samples,
Boundary point: natural neighbors have both majority and minority classes,
}
"""
        nn_k_model = NearestNeighbors(n_neighbors=k+1)
        nn_k_model.fit(self.data)
        neighbors = nn_k_model.kneighbors(self.data, return_distance=False)[:, 1:]  #
        self.neighbors = {i:set(neighbors[i]) for i in range(len(neighbors))}
        self.relative_cox = [0]*len(self.target)    

        
        for i, num in enumerate(neighbors):   
            if self.target[i] == min_i:     
                absolute_min,min_num,absolute_max,maj_num = 0,0,0,0
                maj_index = []

                for j in iter(num): 
                    if self.target[j] == min_i:
                        absolute_min += np.sqrt(np.sum(np.square(self.data[i]-self.data[j])))
                        min_num += 1
                    elif self.target[j] == maj_i:
                        absolute_max += np.sqrt(np.sum(np.square(self.data[i]-self.data[j])))
                        maj_num += 1
                        maj_index.append(j)
                    self.neighbors[i].difference_update(maj_index)    

"""
If 0.7 is counted as noise, the natural neighbors of some minority samples will be empty sets.
This problem has been solved in generate_samples_Lm() in all_smote_v5.py
and SMOTE() in _smote_variants_v5.py
"""

                if min_num == 0 or maj_num >= (min_num + maj_num)*0.7:  # TODO: 
                    self.relative_cox[i] = -3 
                elif maj_num == 0 :         
                    relative = min_num/absolute_min
                    self.relative_cox[i] = relative
                else:           
                    relative = (min_num/absolute_min)/(maj_num/absolute_max)
                    self.relative_cox[i] = relative


def NNk_weight(data_train,k):

"""
knn_weight sampling

return:
------
NData: original data set ndarray
NN.relative_cox: relative density weight ndarray
NN.nan: natural neighbors of each sample; [{},{},...,{}]
"""

    NN = Natural_Neighbor()
    NN.data = data_train[:, 1:]  
    NN.target = data_train[:, 0]
    count = Counter( NN.target )     
    c = count.most_common(len(count))
    min_i,maj_i = c[1][0],c[0][0]
    NN.nn_k(min_i,maj_i,k)

    NN.relative_cox = np.array(NN.relative_cox)      
    NN.neighbors = np.array([list(v) for v in NN.neighbors.values()])    

    return data_train, NN.relative_cox, NN.neighbors    


""" Nan + SMOTE Natural Neighbor Random Sampling"""

import random
from collections import Counter
import numpy as np
from scipy import sparse
from sklearn.utils import check_random_state
from sklearn.utils import _safe_indexing

from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import check_neighbors_object
from imblearn.utils import Substitution
from imblearn.utils._docstring import _n_jobs_docstring
from imblearn.utils._docstring import _random_state_docstring
from imblearn.utils._validation import _deprecate_positional_args


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

    def _make_samples(
        self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0
    ):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Points from which the points will be created.

        y_dtype : dtype.The data type of the targets.
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
        """
        random_state = check_random_state(self.random_state)
        not_null_index = [i for i,nn in enumerate(nn_num) if nn] 
        rows = np.random.choice(not_null_index,n_samples)
        cols = np.array([nn_num[row].index(random.choice(nn_num[row])) for row in rows])
        

        # np.newaxis for backwards compatability with random_state
        steps = step_size * random_state.uniform(size=n_samples)[:, np.newaxis]
        X_new = self._generate_samples(X, nn_data, np.array(nn_num), rows, cols, steps)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new

    def _generate_samples(self, X, nn_data, nn_num, rows, cols, steps):
        """
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
        """
        neighbors = []      
        for i in range(len(rows)):
            neighbors.append(nn_num[rows[i]][cols[i]])

        diffs = nn_data[ neighbors ] - X[rows] 
        if sparse.issparse(X):
            sparse_func = type(X).__name__
            steps = getattr(sparse, sparse_func)(steps)
            X_new = X[rows] + steps.multiply(diffs)
        else:
            X_new = X[rows] + steps * diffs
        return X_new.astype(X.dtype)


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class NaN_SMOTE(BaseSMOTE):

"""
Natural neighbor random sampling
Parameters
----------
k_neighbors: int or object, default=5
"""

    @_deprecate_positional_args
    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
        nans:np.array = None,        
        weight:np.array = None,      
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.nans=nans      
        self.weight=weight  

    def _fit_resample(self, X, y):
        self._validate_estimator()

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:continue

            target_class_indices = np.flatnonzero(self.weight >0)           
            X_class = _safe_indexing(X, target_class_indices)               
            nans = _safe_indexing(self.nans, target_class_indices)          
            
            
            nans_new = [0]*len(X_class)
            for i in range(len(nans)):
                nns = []
                for nn in nans[i]:
                    index = np.where(target_class_indices == nn)
                    if len(index[0]) != 0: 
                        nns.append(index[0][0])
                nans_new[i] = nns

            
            X_new, y_new = self._make_samples(
                X_class, y.dtype, class_sample, X_class, nans_new, n_samples, 1.0
            )
            X_resampled.append(X_new)
            y_resampled.append(y_new)

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:
            X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        return X_resampled, y_resampled
    


'''Nan density weight'''
def generate_samples_zhou_2(X, y_dtype, y_type, nn_data, nn_num, n_samples,
                        weights, danger_and_safe,mother_point=None):
    """
        Parameters
        ----------
        X :         ndarray.   seed samples
        y_dtype :   The data type of the targets.
        y_type :    label of minority.
        nn_data :   ndarray.    Data set carrying all the neighbours to be used
        nn_num :    ndarray of shape (n_samples_all, k_nearest_neighbours)
                    The nearest neighbours of each sample in `nn_data`.
        n_samples : int.    The number of samples to generate.
        weights :   ndarray. The weights.
        danger_and_safe:    int.  The number of seed samples
        mother_point:       ndarray. The index of seed samples

        Returns
        -------
        X_new :     ndarray. synthetically generated samples.
        y_new :     ndarray shape (n_samples_new,). labels for synthetic samples.
    """

    """
# Find the corresponding neighbor points according to the neighbor point index matrix,
# First traverse the seed samples, insert n each seed sample
# Then randomly select m seed samples and insert m
    """
    mother_points = nn_data[mother_point]   
    n = n_samples // danger_and_safe        
    m = n_samples % danger_and_safe         
    weidu = X.shape[1]
    X_new_1 = np.zeros(weidu)
    if not isinstance(nn_data, list):nn_data = nn_data.tolist()


   '''Sort the entire neighbor point index matrix by weight from small to large'''
    nn_num_ordered = []                             
    for i in range(len(nn_num)):                    
        values = weights[nn_num[i]] 
        keys = nn_num[i]   
        dicts,num = {},[]
        for j in range(len(values)):dicts[keys[j]] = values[j]
        d_order=sorted(dicts.items(),key=lambda x:x[1],reverse=False)   
        for d in d_order:num.append(d[0])
        nn_num_ordered.append(num)


   '''N number of points to be inserted for each seed sample'''
    for nn in range(danger_and_safe):               
       # step_1: Get (seed sample) information
        num = nn_num_ordered[nn]                    
        nn_point = mother_points[nn]                
        nn_weight = weights[mother_point[nn]]       
        length_of_num = len(num)                    
        if length_of_num == 0:continue

        for i in range(n):                          
            # step_2: Get neighbor point information
            random_point = num[i%length_of_num]             
            random_point_weight = weights[random_point]     
            random_point_data = nn_data[random_point]       

            # step_3: Start interpolation according to the situation
            proportion = (random_point_weight / (nn_weight + random_point_weight))  
            if nn_weight >= random_point_weight:
                X_new_zhou = np.array(nn_point) + (
                            np.array(random_point_data) - np.array(nn_point)) * proportion * round(
                    random.uniform(0, 1), len(str(n_samples)))
            elif nn_weight < random_point_weight:
                X_new_zhou = np.array(random_point_data) + (np.array(nn_point) - np.array(random_point_data)) * (
                            1 - proportion) * round(random.uniform(0, 1), len(str(n_samples)))
            X_new_1 = np.vstack((X_new_zhou, X_new_1))


    '''Randomly extract m seed samples without replacement'''
    for mm in range(m):
        # step_1: Get seed sample information
        nn_point_index = random.choice(mother_point)        
        nn_point_weight = weights[nn_point_index]           
        nn_point = nn_data[nn_point_index]                  

        # step_2: Get neighbor point information
        num = nn_num[np.where(mother_point == nn_point_index)[0][0]]    
        if num == []:continue
        random_point = num[0]                                           
        random_point_weight = weights[random_point]                     
        random_point_data = nn_data[random_point]                       

        # step_3: Start interpolation
        proportion = (random_point_weight / (nn_point_weight + random_point_weight))  
        if nn_point_weight >= random_point_weight:
            X_new_zhou = np.array(nn_point) + (np.array(random_point_data) - np.array(nn_point)) * proportion * round(random.uniform(0, 1),len(str(n_samples)))
        elif nn_point_weight < random_point_weight:
            X_new_zhou = np.array(random_point_data) + (np.array(nn_point) - np.array(random_point_data)) * (1 - proportion) * round(random.uniform(0, 1), len(str(n_samples)))
        X_new_1 = np.vstack((X_new_zhou, X_new_1))

    X_new_1 = np.delete(X_new_1, -1, 0)

    y_new = np.full(len(X_new_1), fill_value=y_type, dtype=y_dtype)
    return X_new_1.astype(X.dtype) ,y_new


class BaseSMOTE_2(BaseOverSampler):
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


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class Nan_rd_weight(BaseSMOTE_2):
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

            
            X_new, y_new = generate_samples_zhou_2(
                X=X_class,              # seeds: 
                y_dtype=y.dtype,        
                y_type=class_sample,    
                nn_data=X_class,
                nn_num=nans_new,        
                n_samples=n_samples,    
                weights=weight_min,     
                danger_and_safe=len(target_class_indices),              
                mother_point=np.arange(len(target_class_indices)),      
            )
            
            X_resampled.append(X_new)
            y_resampled.append(y_new)

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)
        return X_resampled, y_resampled


'''Nan + number weight: natural neighbor number weight'''
def generate_samples_zhou_3(X, y_dtype, y_type, nn_data, nn_num, n_samples,
                        weights, danger_and_safe,mother_point=None):
    """
        Parameters
        ----------
        X :         ndarray.   seed samples
        y_dtype :   The data type of the targets.
        y_type :    label of minority.
        nn_data :   ndarray.    Data set carrying all the neighbours to be used
        nn_num :    ndarray of shape (n_samples_all, k_nearest_neighbours)
                    The nearest neighbours of each sample in `nn_data`.
        n_samples : int.    The number of samples to generate.
        weights :   ndarray. The weights.
        danger_and_safe:    int.  The number of seed samples
        mother_point:       ndarray. The index of seed samples

        Returns
        -------
        X_new :     ndarray. synthetically generated samples.
        y_new :     ndarray shape (n_samples_new,). labels for synthetic samples.
    """

    """
# Find the corresponding neighbor points according to the neighbor point index matrix,
# First traverse the seed samples, insert n each seed sample
# Then randomly select m seed samples and insert m
    """
    mother_points = nn_data[mother_point]   
    n = n_samples // danger_and_safe        
    m = n_samples % danger_and_safe         
    weidu = X.shape[1]
    X_new_1 = np.zeros(weidu)
    if not isinstance(nn_data, list):nn_data = nn_data.tolist()


  '''Sort the entire neighbor point index matrix by weight from small to large'''
    nn_num_ordered = []                             
    for i in range(len(nn_num)):                    
        values = weights[nn_num[i]] 
        keys = nn_num[i]   
        dicts,num = {},[]
        for j in range(len(values)):dicts[keys[j]] = values[j]
        d_order=sorted(dicts.items(),key=lambda x:x[1],reverse=False)   
        for d in d_order:num.append(d[0])
        nn_num_ordered.append(num)


    '''N number of points to be inserted for each seed sample'''
    for nn in range(danger_and_safe):               
        # step_1: Get (seed sample) information
        num = nn_num_ordered[nn]                    
        nn_point = mother_points[nn]                
        nn_weight = weights[mother_point[nn]]       
        length_of_num = len(num)                    
        if length_of_num == 0:continue

        for i in range(n):                          
            # step_2: Get neighbor point information
            random_point = num[i%length_of_num]             
            random_point_weight = weights[random_point]     
            random_point_data = nn_data[random_point]       


           # step_3: Start interpolation according to the situation
            proportion = (random_point_weight / (nn_weight + random_point_weight))  
            if nn_weight >= random_point_weight:
                X_new_zhou = np.array(nn_point) + (
                            np.array(random_point_data) - np.array(nn_point)) * proportion * round(
                    random.uniform(0, 1), len(str(n_samples)))
            elif nn_weight < random_point_weight:
                X_new_zhou = np.array(random_point_data) + (np.array(nn_point) - np.array(random_point_data)) * (
                            1 - proportion) * round(random.uniform(0, 1), len(str(n_samples)))
            X_new_1 = np.vstack((X_new_zhou, X_new_1))


   '''Randomly extract m seed samples without replacement'''
    for mm in range(m):
        # step_1: Get seed sample information
        nn_point_index = random.choice(mother_point)        
        nn_point_weight = weights[nn_point_index]           
        nn_point = nn_data[nn_point_index]                  

        # step_2: Get neighbor point information
        num = nn_num[np.where(mother_point == nn_point_index)[0][0]]    
        if num == []:continue
        random_point = num[0]                                           
        random_point_weight = weights[random_point]                     
        random_point_data = nn_data[random_point]                       

        # step_3: Start interpolation
        proportion = (random_point_weight / (nn_point_weight + random_point_weight))  
        if nn_point_weight >= random_point_weight:
            X_new_zhou = np.array(nn_point) + (np.array(random_point_data) - np.array(nn_point)) * proportion * round(random.uniform(0, 1),len(str(n_samples)))
        elif nn_point_weight < random_point_weight:
            X_new_zhou = np.array(random_point_data) + (np.array(nn_point) - np.array(random_point_data)) * (1 - proportion) * round(random.uniform(0, 1), len(str(n_samples)))
        X_new_1 = np.vstack((X_new_zhou, X_new_1))

    X_new_1 = np.delete(X_new_1, -1, 0)

    y_new = np.full(len(X_new_1), fill_value=y_type, dtype=y_dtype)
    return X_new_1.astype(X.dtype) ,y_new


class BaseSMOTE_3(BaseOverSampler):
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


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class Nan_num_weight(BaseSMOTE_3):
    """
        k_neighbors : int or object, default=5
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
            # weight_min = _safe_indexing(self.weight, target_class_indices)  
            nans = _safe_indexing(self.nans, target_class_indices)          
            neighbors_num = [len(i) for i in nans]
            neighbors_num_scale = scale(X=neighbors_num,with_mean=True,with_std=True,copy=True) 
            neighbors_num_scale = [abs(i) for i in neighbors_num_scale] 

            
            nans_new = [0]*len(X_class)
            for i in range(len(nans)):
                nns = []
                for nn in nans[i]:
                    index = np.where(target_class_indices == nn)
                    if len(index[0]) != 0: 
                        nns.append(index[0][0])
                nans_new[i] = nns

            
            X_new, y_new = generate_samples_zhou_3(
                X=X_class,              
                y_dtype=y.dtype,        
                y_type=class_sample,    
                nn_data=X_class,        
                nn_num=nans_new,        
                n_samples=n_samples,    
                weights=np.array(neighbors_num_scale),     
                danger_and_safe=len(target_class_indices),              
                mother_point=np.arange(len(target_class_indices)),      
            )
            
            X_resampled.append(X_new)
            y_resampled.append(y_new)

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)
        return X_resampled, y_resampled
    
    
'''Nan + weight(rd + number of Nan) + random number of seeds + rd determines the position weight # WRND seed random version'''
def generate_samples_zhou_4(X, y_dtype, y_type, nn_data, nn_num, n_samples,
                        weights, danger_and_safe,mother_point=None):
    """
        Parameters
        ----------
        X :         ndarray.   seed samples
        y_dtype :   The data type of the targets.
        y_type :    label of minority.
        nn_data :   ndarray.    Data set carrying all the neighbours to be used
        nn_num :    ndarray of shape (n_samples_all, k_nearest_neighbours)
                    The nearest neighbours of each sample in `nn_data`.
        n_samples : int.    The number of samples to generate.
        weights :   ndarray. The weights.
        danger_and_safe:    int.  The number of seed samples
        mother_point:       ndarray. The index of seed samples

        Returns
        -------
        X_new :     ndarray. synthetically generated samples.
        y_new :     ndarray shape (n_samples_new,). labels for synthetic samples.
    """

    """
# Find the corresponding neighbor points according to the neighbor point index matrix,
# Each seed synthesizes a random number of samples
    """
    mother_points = nn_data[mother_point]   
    # n = n_samples // danger_and_safe        
    # m = n_samples % danger_and_safe         

    
    seed_float_rand = np.random.rand(danger_and_safe)    
    ratio = n_samples/sum(seed_float_rand)
    seed_float_rand = seed_float_rand * ratio
    seed_int_rand = seed_float_rand.astype(int)
    print(seed_int_rand)
    print(sum(seed_int_rand),n_samples)

    weidu = X.shape[1]
    X_new_1 = np.zeros(weidu)
    if not isinstance(nn_data, list):nn_data = nn_data.tolist()


  '''Sort the entire neighbor point index matrix by weight from small to large'''
    nn_num_ordered = []                             
    for i in range(len(nn_num)):                    
        values = weights[nn_num[i]] 
        keys = nn_num[i]   
        dicts,num = {},[]
        for j in range(len(values)):dicts[keys[j]] = values[j]
        d_order=sorted(dicts.items(),key=lambda x:x[1],reverse=False)   
        for d in d_order:num.append(d[0])
        nn_num_ordered.append(num)


    
    for nn in range(danger_and_safe):               
        # step_1: Get (seed sample) information
        num = nn_num_ordered[nn]                    
        nn_point = mother_points[nn]                
        nn_weight = weights[mother_point[nn]]       
        length_of_num = len(num)                    
        if length_of_num == 0:continue

        for i in range(seed_int_rand[nn]):                          
            # step_2: Get neighbor point information
            random_point = num[i%length_of_num]             
            random_point_weight = weights[random_point]     
            random_point_data = nn_data[random_point]       

            # step_3: Start interpolation according to the situation
            proportion = (random_point_weight / (nn_weight + random_point_weight))  
            if nn_weight >= random_point_weight:
                X_new_zhou = np.array(nn_point) + (
                            np.array(random_point_data) - np.array(nn_point)) * proportion * round(
                    random.uniform(0, 1), len(str(n_samples)))
            elif nn_weight < random_point_weight:
                X_new_zhou = np.array(random_point_data) + (np.array(nn_point) - np.array(random_point_data)) * (
                            1 - proportion) * round(random.uniform(0, 1), len(str(n_samples)))
            X_new_1 = np.vstack((X_new_zhou, X_new_1))


'''N points to be interpolated for each seed sample'''
# for nn in range(danger_and_safe): # Traverse each seed sample
# # step_1: Get (seed sample) information
# num = nn_num_ordered[nn] # Neighbor point index matrix of seed sample
# nn_point = mother_points[nn] # Seed sample features
# nn_weight = weights[mother_point[nn]] # Seed sample weight
# length_of_num = len(num) # Length of neighbor point matrix
# if length_of_num == 0:continue

# for i in range(n): # Interpolate n times based on seed sample
# # step_2: Get neighbor point information
# random_point = num[i%length_of_num] # Extract neighbor point index from small to large according to weight
# random_point_weight = weights[random_point] # Neighbor point weight
# random_point_data = nn_data[random_point] # Features of neighboring points

# # step_3: Start interpolation according to the situation
# proportion = (random_point_weight / (nn_weight + random_point_weight)) # Weight ratio
# if nn_weight >= random_point_weight:
# X_new_zhou = np.array(nn_point) + (
# np.array(random_point_data) - np.array(nn_point)) * proportion * round(
# random.uniform(0, 1), len(str(n_samples)))
# elif nn_weight < random_point_weight:
# X_new_zhou = np.array(random_point_data) + (np.array(nn_point) - np.array(random_point_data)) * (
# 1 - proportion) * round(random.uniform(0, 1), len(str(n_samples)))
# X_new_1 = np.vstack((X_new_zhou, X_new_1))

'''Randomly extract m seed samples without replacement'''
# for mm in range(m):
# # step_1: Get seed sample information
# nn_point_index = random.choice(mother_point) # Index of seed sample
# nn_point_weight = weights[nn_point_index] # Weight of seed sample
# nn_point = nn_data[nn_point_index] # Features of seed sample

# # step_2: Get neighbor point information
# num = nn_num[np.where(mother_point == nn_point_index)[0][0]] # Neighbor index list # Neighbor point list of seed sample
# if num == []:continue
# random_point = num[0] # Neighbor point index
# random_point_weight = weights[random_point] # Neighbor point weight
# random_point_data = nn_data[random_point] # Neighbor point features

# # step_3: Start interpolation
# proportion = (random_point_weight / (nn_point_weight + random_point_weight)) # Weight ratio
# if nn_point_weight >= random_point_weight:
# X_new_zhou = np.array(nn_point) + (np.array(random_point_data) - np.array(nn_point)) * proportion * round(random.uniform(0, 1),len(str(n_samples)))
# elif nn_point_weight < random_point_weight:
# X_new_zhou = np.array(random_point_data) + (np.array(nn_point) - np.array(random_point_data)) * (1 - proportion) * round(random.uniform(0, 1), len(str(n_samples))) # X_new_1 = np.vstack((X_new_zhou, X_new_1))

    X_new_1 = np.delete(X_new_1, -1, 0)

    y_new = np.full(len(X_new_1), fill_value=y_type, dtype=y_dtype)
    return X_new_1.astype(X.dtype) ,y_new




class BaseSMOTE_4(BaseOverSampler):
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


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class Nan_weight_rand(BaseSMOTE_4):
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

            
            X_new, y_new = generate_samples_zhou_4(
                X=X_class,              # seeds: 
                y_dtype=y.dtype,        
                y_type=class_sample,    
                nn_data=X_class,        
                nn_num=nans_new,        
                n_samples=n_samples,    
                weights=weight_min,     
                danger_and_safe=len(target_class_indices),              
                mother_point=np.arange(len(target_class_indices)),      
            )
            
            X_resampled.append(X_new)
            y_resampled.append(y_new)

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)
        return X_resampled, y_resampled


