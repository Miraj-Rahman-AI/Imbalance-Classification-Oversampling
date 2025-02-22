import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter

class ASmote:
    """
    class ASmote usage is as follows:
    "
    data_Asmote = AdaptiveSMOTE.ASmote(data)
    .over_sampling()
    "
    """

    def __init__(self, data,  random_state=None):
        """
        :param data: array for all data with label in 0th col.
        :param ir: imbalanced ratio of synthetic data.
        :param k: Number of nearest neighbors.
        """
        self.data = data    # 标签在第一列

    def dataReset(self):
        """
        X, Y: 多+少 numpy.ndarray
        Xn:多数类数据列
        Xp: 少数类数据列
        tp_more,tp_less:多数类与少数类标签列
        划分多数类与少数类
        """
        # 统计正、负类样本并拆分开
        count = Counter(self.data[:, 0])
        if len(count) != 2:
            raise Exception('数据集 {} 标签类别数不为2： {}'.format(data_name, count))
        tp_more, tp_less = set(count.keys())
        # print(tp_more, tp_less)
        if count[tp_more] < count[tp_less]:
            tp_more, tp_less = tp_less, tp_more
        data_more = self.data[self.data[:, 0] == tp_more]
        data_less = self.data[self.data[:, 0] == tp_less]
        # print(self.data.shape, data_more.shape, data_less.shape)
        X = self.data[:,1:]
        Xn = data_more[:,1:]
        Xp = data_less[:,1:]
        # print(X.shape, Xn.shape, Xp.shape)
        # print(data_more[0], data_less[0])

        return X, Xn,tp_more, Xp, tp_less

def oversampleASmote(data, N, Xp):
    """
    imput: data 多+少  N多数类  Xp少数类
    # Nnum: 多数类数量 Pnum:少数类数量   dim: 数据维度
    numpy.ndarray 
    return: data:data+Xnew Xp:Xp+Xnew
    """
    Nnum = N.shape[0]    # 多数类数量
    Pnum = Xp.shape[0]   # 少数类数量
    dim = N.shape[1]     # 数据维度
    if Pnum <= 1.05*Nnum:
        (boundray, inner) = boundray_inner(data, Xp, Nnum, dim)         # boundray_inner
        if len(boundray) > 1 and len(inner) > 1:
            new = new1_data(data, Xp, inner, boundray, Nnum, dim)[1:]

        elif Pnum == 0:
            new = random_SMOTE(Xp, int(round(Nnum)))               # 随机采样
        else:
            new = random_SMOTE(Xp, int(round(Nnum/Pnum)))               # 随机采样
        Xp = np.vstack((Xp, new))
        data = np.vstack((data, new))
        if len(Xp) < Nnum:
            if len(Xp) == 0:
                new1 = random_SMOTE(Xp, int(round(Nnum)))           # 随机采样
            else:
                new1 = random_SMOTE(Xp, int(round(Nnum/len(Xp))))           # 随机采样
            Xp = np.vstack((Xp, new1))
            data = np.vstack((data, new1))
    return Xp
    
# [boundray,inner]=boundray_inner(data,P,Nnum,b)
# P include inner data and boundray dara merely.
# boundray0, inner0: started with 0 vector.
# size(boundray0)+size(inner0)=size(P)+2
def boundray_inner(X, X_1, Nnum, b):
    inner0 = np.zeros((1, b+1))
    boundray0 = np.zeros((1, b+1))
    r, c = X_1.shape
    for i in range(0, r):
        inner = False
        for k in range(5, 11):
            if np.sum(neigh0(X, X_1, k, i)>= Nnum) > k/2:
                inner = True
        if inner:
            inner0 = np.vstack((inner0, np.hstack((i, X_1[i]))))
        else:
            boundray0 = np.vstack((boundray0, np.hstack((i, X_1[i]))))
    return(boundray0, inner0)

def new1_data(data, P, inner, boundray, Nnum, b):
    Pnew = np.zeros((1, b))
    for i in range(1, len(inner)):
        [nn] = neigh0(data, inner[:, 1:], 10, i)
        AND = set(list(nn-Nnum)).intersection(set(list(boundray[1:, 0])))
        AND = list(AND)
        for j in range(len(AND)):
            AND[j] = int(AND[j])
        if not(AND == set([])):
            for j in AND:
                dif = P[j]-inner[i, 1:]
                Pnew = np.vstack((Pnew, inner[i, 1:]+np.random.rand()*dif))
    return(Pnew)
def neigh0(data, P, k, i):  # k-nn
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(data)
    nn = neigh.kneighbors(P[i].reshape(1, -1), return_distance=False)
    return nn

def random_SMOTE(data, N):
    """ random-SMOTE 

    Args: 
         data: numpy.ndarray, positive data, dimension: (number of data x number of feature) 
         N: int, multiple of oversample 

    Returns: 
        numpy.ndarray, generate new data of positive 
    """

    sample_num, feature_dim = data.shape

    new_data = np.zeros((sample_num*N, feature_dim))  # new data of positive
    tmp_data = np.zeros((N, feature_dim))   # temporary data

    for i in range(sample_num):
        X = data[i]
        # the first random index eiliminate X.
        idx1 = _rand_idx(0, sample_num - 1, (i, ))
        # the second random index eliminate X and first index.
        idx2 = _rand_idx(0, sample_num - 1, (i, idx1))
        Y1 = data[idx1]
        Y2 = data[idx2]

        # generate new temporary data
        for j in range(N):
            for k in range(feature_dim):
                dif = Y2[k] - Y1[k]
                tmp_data[j][k] = Y1[k] + dif * np.random.rand()

        # generate new data
        for j in range(N):
            for k in range(feature_dim):
                dif = tmp_data[j][k] - X[k]
                new_data[i * N + j][k] = X[k] + dif * np.random.rand()

    return new_data 

def _rand_idx(start, end, exclude=None):
    """ generate random index, index eliminate 'exclude' """
    # print(start, end)
    if start > end:
        start, end = end, start
    elif start == end:
        rev = start
        return rev
    rev = np.random.randint(start, end)
    if exclude is None:
        return rev
    else:
        while rev in exclude:
            rev = np.random.randint(start, end)
        return rev