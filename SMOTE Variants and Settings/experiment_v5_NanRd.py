import _imblearn_v5_NanRd_v2 as _imblearn_v5_NanRd_v2
import _smote_variants_v5_NanRd_v2 as _smote_variants_v5_NanRd_v2
import _smote_variants_v1_orignal as _smote_variants_v1_orignal
from sklearn.ensemble import (
    GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score,
                             f1_score, roc_curve, precision_score, confusion_matrix, precision_recall_fscore_support,precision_recall_curve,average_precision_score)

from RSMOTE import RSmote, RSmoteKClasses
from AdaptiveSMOTE import ASmote, oversampleASmote
from ldas import LDAS
# from gsmote import GeometricSMOTE
from TKDE_GDO import GDO
from ZhengSY import main
import _smote_variants_v7_wkb
import _smote_variants_gbsmote

import os
import subprocess
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn import over_sampling
from sklearn.model_selection import KFold
import time
from xlutils.copy import copy
import xlrd
import xlwt
from collections import Counter
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random
import warnings
warnings.filterwarnings('ignore')



def now_time_str(colon=True):
    t = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    if not colon:
        t = t.replace(':', '')
    return t


def mean_std(a):
    """
    Calculate the mean and standard deviation of a list.
    :param a: list
    :return: mean and standard
    """
    a = np.array(a)
    std = np.sqrt(((a - np.mean(a)) ** 2).sum() / (a.size - 1))
    return a.mean(), std


def add_noise(data, noise_rate, random_state=1):
    """
Randomly flip the labels of samples with the number of noise_rate * n. The default label has only two categories. 
:param data: binary classification data, the first column is the label. 
:param noise_rate: the noise rate to be added. 
:param random_state: random seed. 
:return: None
    """
    random.seed(random_state)
    labels = list(set(data[:, 0]))
    if len(labels) == 1:
        swap_label = {0: 1, 1: 0}
    else:
        swap_label = {labels[0]: labels[1], labels[1]: labels[0]}

    n = data.shape[0]
    noise_num = int(n * noise_rate)
    index = list(range(n))
    random.shuffle(index)

    # data_noise = data[index[:noise_num], :]
    # data_noise[:, 0] = np.array(list(map(lambda x: swap_label[x], data_noise[:, 0])))
    for i in index[:noise_num]:
        data[i, 0] = swap_label[data[i, 0]]


def add_noise_lx(dataset, noise_rate, random_state=1):
    random.seed(random_state)
    label_cat = sorted(list(set(dataset[:, 0])))  
    new_data = np.array([])
    flag = 0
    for i in range(len(label_cat)):
        label = label_cat[i]
        other_label = list(filter(lambda x: x != label, label_cat))
        data = dataset[dataset[:, 0] == label]
        n = data.shape[0]
        noise_num = int(n * noise_rate)
        noise_index_list = []  
        n_index = 0
        while True:
            rand_index = int(random.uniform(0, n))  
            if rand_index in noise_index_list:  
                continue

            if n_index < noise_num:  
                data[rand_index, 0] = random.choice(other_label)  
                n_index += 1
                noise_index_list.append(rand_index)

            if n_index >= noise_num:
                break
        if flag == 0:
            new_data = data
            flag = 1
        else:
            new_data = np.vstack([new_data, data])
    return new_data


def split_data_set(data, proportion, random_state=2):
    """
Randomly divide data into several parts
:param data:
:param proportion: The proportion of the division, for example, if proportion = [0.8, 0.2], then two parts of 80% and 20% are returned
:param random_state: Random seed
:return:
    """
    assert sum(proportion) == 1
    random.seed(random_state)

    n = data.shape[0]
    index = list(range(n))
    random.shuffle(index)

    data_ret = []
    count_n = 0
    for i in range(len(proportion)):
        n = int(sum(proportion[:i + 1]) * data.shape[0])
        data_ret.append(data[index[count_n: count_n + n], :])
        count_n += n
    return data_ret


def random_sampling(data, sampling_num, random_state=3):
    """
Simple random sampling
:param data: data
:param sampling_num: number of samples
:param random_state: random seed
:return:
    """
    random.seed(random_state)
    n = data.shape[0]
    index = list(range(n))
    random.shuffle(index)
    index_sampling = sorted(index[:sampling_num])
    return data[index_sampling, :]


def plot_data_and_balls(data_original, data_sampling, center=(), radius=(), ball_labels=(), color_dark=0.1,
                        title='', x_label='', y_label='', file_path='',
                        image_format=None, show_pic=True, close=True):
    """
Draw 2D data points and spheres
:param data_original:
:param data_sampling:
:param center:
:param radius:
:param color_dark: the color of data_original, the closer to 0, the lighter the color
:param title:
:param x_label:
:param y_label:
:param file_path:
:param image_format: '.svg', '.eps', '.png', '.jpg'
:param show_pic:
:param close:
:return:
    """
    data0 = data_original[data_original[:, 0] != 1]
    data1 = data_original[data_original[:, 0] == 1]
    c = 1 - color_dark
    plt.plot(data0[:, 1], data0[:, 2], '.', color=(c, c, c), markersize=3)
    plt.plot(data1[:, 1], data1[:, 2], '.', color=(1, c, c), markersize=3)

    data0 = data_sampling[data_sampling[:, 0] != 1]
    data1 = data_sampling[data_sampling[:, 0] == 1]
    # plt.rcParams['figure.figsize'] = (8.0, 8.0)  
    plt.plot(data0[:, 1], data0[:, 2], '.k')
    plt.plot(data1[:, 1], data1[:, 2], '.r')

    if center:
        for i in range(len(center)):
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = center[i][0] + radius[i] * np.cos(theta)
            y = center[i][1] + radius[i] * np.sin(theta)
            color = 'k' if ball_labels[i] == 0 else 'r'
            plt.plot(x, y, color, linewidth=0.4)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axis('equal')

    # plt.axis([-2, 3, -1.5, 2])  # make_moons
    # plt.axis([-2, 2, -2, 2])  # make_circles
    # plt.grid(linestyle='-', color='#D3D3D3')  

    if show_pic:
        plt.show()
    else:
        if isinstance(image_format, str):
            image_format = [image_format]
        for img in image_format:
            if img in ('.jpg', '.png'):
                plt.savefig(file_path + img, dpi=1000, bbox_inches='tight')
            elif img == '.emf':
                fig = plt.gcf()
                plot_as_emf(fig, filename=file_path + img)
            else:
                plt.savefig(file_path + img)

    if close:
        plt.close()


def plot_hyperplane(clf, X, y,
                    title='', x_label='', y_label='', file_path='', image_format=(), show_pic=True, close=True):

    clf.fit(X, y)
    x_min, x_max = X[:, 0].min() - 0.05, X[:, 0].max() + 0.05
    y_min, y_max = X[:, 1].min() - 0.05, X[:, 1].max() + 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.002),
                         np.arange(y_min, y_max, 0.002))
    # predict the point
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    # red yellow blue

    plt.contourf(xx, yy, Z, cmap='Pastel1')

    colors = ['k', 'r']
    labels = [0, 1]

    for label in [0, 1]:
        plt.scatter(X[y == labels[label], 0], X[y == labels[label],
                                                1], c=colors[label], s=6, cmap=plt.cm.RdYlBu)

    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axis('equal')

    # fig = plt.gcf()
    #     if draw_sv:
    #         sv = clf.support_vectors_
    #         plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')

    if show_pic:
        plt.show()
    else:
        for img in image_format:
            if img in ('.jpg', '.png'):
                plt.savefig(file_path + img, dpi=1000, bbox_inches='tight')
            elif img == '.emf':
                fig = plt.gcf()
                plot_as_emf(fig, filename=file_path + img)
            else:
                plt.savefig(file_path + img)

    if close:
        plt.close()


def plot_as_emf(figure, **kwargs):
    """
    python emf 
    http://blog.sciencenet.cn/home.php?mod=space&uid=730445&do=blog&quickforward=1&id=1196366
    """
    inkscape_path = kwargs.get(
        "inkscape", "C:\Program Files\Inkscape\inkscape.exe")
    filepath = kwargs.get('filename', None)

    if filepath is not None:
        path, filename = os.path.split(filepath)
        filename, extension = os.path.splitext(filename)

        svg_filepath = os.path.join(path, filename + '.svg')
        emf_filepath = os.path.join(path, filename + '.emf')

        figure.savefig(svg_filepath, format='svg')

        subprocess.call([inkscape_path, svg_filepath,
                         '--export-emf', emf_filepath])
        os.remove(svg_filepath)


def load_2classes_data(data_name, proportion=(0.8, 0.2), noise_rate=0.0, imbalance_rate=0, normalized=True, n_rows=0,
                       random_state=0, add_noise_way='together'):
    """
Read 2-class dataset by dataset name
:param data_name:
:param proportion: training set, test set ratio
:param noise_rate: noise rate, add noise only in training set
:param imbalance_rate: treat as imbalanced data according to this imbalance ratio, default value 0 means no processing
:param normalized: whether to normalize
:param n_rows: read the number of rows, if 0, read all rows, otherwise read the specified number of rows randomly
:param add_noise_way: 'separately' : positive and negative samples are adjusted for noise separately, the number of noise is minority class samples * noise_rate,
'together' : positive and negative samples are combined and noise is added randomly together
:return: training set, test set data, column 0 is the label, label is 0, 1
counters = (samples, features, train0, train1, test0, test1)
    """

    data16 = ['fourclass', 'svmguide1', 'diabetes', 'codrna', 'breastcancer', 'creditApproval', 'votes', 'ijcnn1',
              'svmguide3', 'sonar', 'splice', 'mushrooms', 'clean1', 'madelon_train', 'madelon_test', 'isolet5',
              'isolet1234']
    data_big1 = ['avila', 'letter', 'susy']
    data_big2 = ['mocap', 'poker', 'nomao']
    data_big3 = ['magic', 'skin', 'covtype', 'comedy', 'Online_Retail']
    data_big = data_big1 + data_big2 + data_big3
    uci_extended = ['breast_tissue2', 'ecoli', 'glass', 'haberman', 'heart', 'iris', 'libra', 'liver_disorders',
                    'pima', 'segment', 'vehicle', 'wine', 'simulated1', 'simulated2', 'simulated3', 'simulated4',
                    'simulated5', 'simulated6', 'simulated7', 'simulated8', 'simulated9', 'simulated10']  
    uci_extended_fast = ['breast_tissue2', 'ecoli', 'glass', 'haberman', 'heart', 'iris', 'libra', 'liver_disorders',
                         'pima', 'segment', 'vehicle', 'wine']
    uci_extended_simulated = ['simulated1', 'simulated2', 'simulated3', 'simulated4', 'simulated5', 'simulated6',
                              'simulated7', 'simulated8', 'simulated9', 'simulated10']
    uci_extended_not_exist = ['new-thyroid1', 'new-thyroid2', 'cleveland-0', 'dermatology-6', 'led7digit',
                              'page-blocks0', 'page-blocks-1-3', 'vowel0', 'yeast1', 'yeast2', 'yeast3', 'yeast4',
                              'yeast5', 'yeast-0-2-5-6', 'yeast-0-2-5-7-9', 'yeast-0-3-5-9', 'yeast-0-5-6-7-9',
                              'yeast-1-2-8-9', 'yeast-2_vs_4', 'yeast-2_vs_8', ]
    data_small = ['anneal', 'credit', 'german', 'heart1', 'heart2', 'hepatitis', 'horse', 'iono', 'sonar', 'wdbc',
                  'wine', 'letter', 'lymphography', 'mushroom', 'soybean', 'zoo']
    data_small_2_class = ['credit', 'german',
                          'heart1', 'hepatitis', 'horse', 'iono', 'wdbc']
    data_pinjie = ['madelon', 'isolet']
    data_gongye = ['Data_for_UCI_named', 'default of credit card clients', 'diabetic',
                   'Epileptic Seizure Recognition', 'Faults', 'HCV-Egy-Data',
                   'messidor_features', 'OnlineNewsPopularity', 'sat', 'seismic-bumps',
                   'shuttle', 'wilt']
    # KEEL imb_IRlowerThan9
    imb_IRlowerThan9 = ['ecoli-0_vs_1', 'ecoli1', 'ecoli2', 'ecoli3', 'glass0', 'glass-0-1-2-3_vs_4-5-6', 'glass1', 'glass6',
                        'haberman', 'iris0', 'new-thyroid1', 'new-thyroid2', 'page-blocks0', 'pima', 'segment0', 'vehicle0',
                        'vehicle1', 'vehicle2', 'vehicle3', 'wisconsin', 'yeast1', 'yeast3']
    # imb_extended = ['glass0', 'new-thyroid1','yeast1',  'page-blocks0',] 
    imb_extended = ['new-thyroid1', 'page-blocks0', ]

    assert data_name in uci_extended + data16 + data_big + data_small_2_class + data_pinjie+data_gongye+imb_IRlowerThan9, 'data set \'{}\' is not exist!'.format(
        data_name)
    assert add_noise_way in ('separately', 'together')

    experiment_path = r'KBsomte/'  

    if data_name in uci_extended:
        df = pd.read_csv(experiment_path +
                         r'/SMOTE/uci_extended/' + data_name + '.csv')
        data = df.values
        data = np.hstack((data[:, -1:], data[:, :-1]))
    elif data_name in data_gongye:
        df = pd.read_csv(experiment_path +
                         r'/data_set/digital_twin/' + data_name + '.csv')
        data = df.values
    elif data_name in data_big:
        df = pd.read_csv(
            experiment_path + r'/data_set/large_data/' + data_name + '.csv', header=None)
        data = df.values
    elif data_name in data_small:
        df = pd.read_csv(
            experiment_path + r'/data_set/small_data/' + data_name + '.csv', header=None)
        data = df.values
    # KEEL imb_IRlowerThan9
    elif data_name in imb_IRlowerThan9:
        df = pd.read_csv(
            experiment_path + r'/data_set/imb_IRlowerThan9/' + data_name + '.csv', header=None)
        data = df.values
        data = np.hstack((data[:, -1:], data[:, :-1]))
    else:
        data_mat = scipy.io.loadmat(
            experiment_path + r'/data_set/dataset16/dataset16.mat')
        if data_name == 'madelon':
            data = np.vstack(
                (data_mat['madelon_train'], data_mat['madelon_test']))
        elif data_name == 'isolet':
            data = np.vstack((data_mat['isolet1234'], data_mat['isolet5']))
        else:
            data = data_mat[data_name]

    np.random.seed(random_state + 4)
    np.random.shuffle(data)
    if 0 < n_rows < data.shape[0]:
        data = data[:n_rows]

    
    data = data.astype(np.float64)
    if normalized:
        for f in range(1, data.shape[1]):
            ma, mi = max(data[:, f]), min(data[:, f])
            subtract = ma - mi
            if subtract != 0:
                data[:, f] = (data[:, f] - mi) / subtract

    
    count = Counter(data[:, 0])
    if len(count) != 2:
        raise Exception(' {} 2： {}'.format(data_name, count))
    tp_more, tp_less = set(count.keys())
    if count[tp_more] < count[tp_less]:
        tp_more, tp_less = tp_less, tp_more
    data_more = data[data[:, 0] == tp_more]
    data_less = data[data[:, 0] == tp_less]

    
    data_more[:, 0] = 0
    data_less[:, 0] = 1
    tp_more, tp_less = 0, 1

    if imbalance_rate != 0 and (data_name not in uci_extended and data_name not in imb_extended):
        assert imbalance_rate > 1
        data_less_num = int(data_more.shape[0] // imbalance_rate)
        data_less = random_sampling(
            data_less, data_less_num, random_state=random_state)


    data_more_train, data_more_test = split_data_set(
        data_more, proportion, random_state=random_state)
    data_less_train, data_less_test = split_data_set(
        data_less, proportion, random_state=random_state)

    if add_noise_way == 'separately':
        add_noise(data_more_train, noise_rate=noise_rate * data_less_train.shape[0] / data_more_train.shape[0],
                  random_state=random_state)
        add_noise(data_less_train, noise_rate=noise_rate,
                  random_state=random_state)


    data_train = np.vstack((data_more_train, data_less_train))
    data_test = np.vstack((data_more_test, data_less_test))

    if add_noise_way == 'together':
        add_noise(data_train, noise_rate=noise_rate, random_state=random_state)

    np.random.seed(random_state + 4)
    np.random.shuffle(data_train)
    np.random.shuffle(data_test)

    # counters = (samples, features, train0, train1, test0, test1)
    counters = (data_train.shape[0] + data_test.shape[0], data_train.shape[1] - 1, Counter(data_train[:, 0])[0],
                Counter(data_train[:, 0])[1], Counter(data_test[:, 0])[0], Counter(data_test[:, 0])[1], count)

    return data_train, data_test, counters


def load_k_classes_data_lm(data_name, proportion=(0.8, 0.2), noise_rate=0.0, normalized=True, n_rows=0,
                           random_state=0, add_noise_way='together'):
    data_small = ['anneal', 'credit', 'german', 'heart1', 'heart2', 'hepatitis', 'horse', 'iono', 'sonar', 'wdbc',
                  'wine', 'lymphography', 'mushroom', 'soybean', 'zoo']
    data_zy = ['abalone', 'balancescale', 'car', 'contraceptive', 'ecoli', 'fourclass', 'frogs', 'glass', 'iris',
               'letter', 'newthyroid', 'nuclear', 'OBS', 'pendigits', 'PhishingData', 'poker', 'satimage', 'seeds',
               'segmentation', 'sensorReadings', 'shuttle', 'svmguide2', 'svmguide4', 'userknowledge', 'vehicle',
               'vertebralColumn', 'vowel', 'wifiLocalization', 'yeast', 'krkopt', 'shuttle_all',
               'Healthy_Older_People2']

    data_all = data_small + data_zy

    if len(set(data_all)) != len(data_all):  
        d = filter(lambda x: x[1] != 1, Counter(data_all).items())
        raise Exception('：{}'.format(list(d)[0][0]))

    experiment_path = r'KBsomte/'  

    if data_name in data_small:
        df = pd.read_csv(experiment_path +
                         r'data_set/small_data/' + data_name + '.csv', header=None)
    
        data = df.values
    elif data_name in data_zy:
        df = pd.read_csv(experiment_path +
                         r'data_set/DataSet/' + data_name + '.csv', header=None)

        data = df.values
    else:
        assert 0 == 1, 'data set \'{}\' is not exist!'.format(data_name)

    np.random.seed(random_state + 4)
    np.random.shuffle(data)
    if 0 < n_rows < data.shape[0]:
        data = data[:n_rows]
    if normalized:
        for f in range(1, data.shape[1]):
            ma, mi = max(data[:, f]), min(data[:, f])
            subtract = ma - mi
            if subtract != 0:
                data[:, f] = (data[:, f] - mi) / subtract

    count = Counter(data[:, 0])
    print(len(count))
    count_dict = dict(count)    
    # print(type(count_dict), count_dict)
    count_sorted_values = sorted(
        count_dict.items(), key=lambda x: x[1], reverse=True)  
    # print(count_sorted_values)
    # print(count_sorted_values[-1][0], count_sorted_values[0][0])
    tp_less, tp_more = count_sorted_values[-1][0], count_sorted_values[0][0]
    data_more = data[data[:, 0] != tp_less]
    data_less = data[data[:, 0] == tp_less]
    # print(len(data_more), len(data_less))

    data_more[:, 0] = 0
    data_less[:, 0] = 1
    tp_more, tp_less = 0, 1
    data_more_train, data_more_test = split_data_set(
        data_more, proportion, random_state=random_state)
    data_less_train, data_less_test = split_data_set(
        data_less, proportion, random_state=random_state)

    if add_noise_way == 'separately':
        add_noise(data_more_train, noise_rate=noise_rate * data_less_train.shape[0] / data_more_train.shape[0],
                  random_state=random_state)
        add_noise(data_less_train, noise_rate=noise_rate,
                  random_state=random_state)

    data_train = np.vstack((data_more_train, data_less_train))
    data_test = np.vstack((data_more_test, data_less_test))

    if add_noise_way == 'together':
        add_noise(data_train, noise_rate=noise_rate, random_state=random_state)

    np.random.seed(random_state + 4)
    np.random.shuffle(data_train)
    np.random.shuffle(data_test)

    # counters = (samples, features, train0, train1, test0, test1)
    counters = (data_train.shape[0] + data_test.shape[0], data_train.shape[1] - 1, Counter(data_train[:, 0])[0],
                Counter(data_train[:, 0])[1], Counter(data_test[:, 0])[0], Counter(data_test[:, 0])[1], count)

    return data_train, data_test, counters


def load_k_classes_data(data_name, noise_rate=0.0,  proportion=(0.8, 0.2), normalized=True, random_state=0, n_rows=0):
    data_small = ['anneal', 'credit', 'german', 'heart1', 'heart2', 'hepatitis', 'horse', 'iono', 'sonar', 'wdbc','wine', 'lymphography', 'mushroom', 'soybean', 'zoo']
    data_zy = ['abalone', 'balancescale', 'car', 'contraceptive', 'ecoli', 'fourclass', 'frogs', 'glass', 'iris',
               'letter', 'newthyroid', 'nuclear', 'OBS', 'pendigits', 'PhishingData', 'poker', 'satimage', 'seeds',
               'segmentation', 'sensorReadings', 'shuttle', 'svmguide2', 'svmguide4', 'userknowledge', 'vehicle',
               'vertebralColumn', 'vowel', 'wifiLocalization', 'yeast', 'krkopt', 'shuttle_all',
               'Healthy_Older_People2']

    data_all = data_small + data_zy

    if len(set(data_all)) != len(data_all):  
        d = filter(lambda x: x[1] != 1, Counter(data_all).items())
        raise Exception('：{}'.format(list(d)[0][0]))

    experiment_path = r'KBsomte/'  

    if data_name in data_small:
        df = pd.read_csv(experiment_path +
                         r'data_set/small_data/' + data_name + '.csv', header=None)

        data = df.values
    elif data_name in data_zy:
        df = pd.read_csv(experiment_path +
                         r'data_set/DataSet/' + data_name + '.csv', header=None)

        data = df.values
    else:
        assert 0 == 1, 'data set \'{}\' is not exist!'.format(data_name)

    np.random.seed(random_state + 4)
    np.random.shuffle(data)
    if 0 < n_rows < data.shape[0]:
        data = data[:n_rows]

    
    if normalized:
        for f in range(1, data.shape[1]):
            ma, mi = max(data[:, f]), min(data[:, f])
            subtract = ma - mi
            if subtract != 0:
                data[:, f] = (data[:, f] - mi) / subtract

    data_train = np.array([]).reshape((-1, data.shape[1]))
    data_test = np.array([]).reshape((-1, data.shape[1]))
    labels = list(set(data[:, 0]))
    for label in labels:
        train, test = split_data_set(
            data[data[:, 0] == label, :], proportion, random_state=random_state)
        data_train = np.vstack((data_train, train))
        data_test = np.vstack((data_test, test))

    data_train = add_noise_lx(
        data_train, noise_rate=noise_rate, random_state=random_state)

    np.random.seed(random_state + 4)
    np.random.shuffle(data_train)
    np.random.shuffle(data_test)

    # counters = (samples, features, train0, train1, test0, test1)
    count = Counter(data[:, 0])
    counters = (data_train.shape[0] + data_test.shape[0], data_train.shape[1] - 1, Counter(data_train[:, 0])[0],
                Counter(data_train[:, 0])[1], Counter(data_test[:, 0])[0], Counter(data_test[:, 0])[1], count)
    # print('count',count)

    return data_train, data_test, counters


def get_metrics(clf, data_train, data_test):
    if len(Counter(data_train[:, 0])) < 2:
        raise ('{} Only one class in the data set {}'.format('  ' * 10, '!' * 10))
    #  NaN/infinity/too large value
    if np.isnan(data_train).any() == True:
        data_train = data_train[~np.isnan(data_train).any(axis=1)]  
    if np.isnan(data_test).any() == True:
        data_test = data_test[~np.isnan(data_test).any(axis=1)]  
    # binary class
    if len(Counter(data_train[:, 0])) == 2:
        clf.fit(data_train[:, 1:], data_train[:, 0])
        predict = clf.predict(data_test[:, 1:])
        predict_proba = clf.predict_proba(data_test[:, 1:])[:, 1]

        accuracy = accuracy_score(data_test[:, 0], predict)  
        precision = precision_score(data_test[:, 0], predict)
        recall = recall_score(data_test[:, 0], predict)
        auc = roc_auc_score(data_test[:, 0], predict_proba)  
        f1 = f1_score(data_test[:, 0], predict)
        tn, fp, fn, tp = confusion_matrix(
            data_test[:, 0], predict).ravel()  
        g_mean = math.sqrt(recall*(tn/(tn+fp)))
        
        
        ap = average_precision_score(data_test[:, 0], predict_proba, average='macro', pos_label=1, sample_weight=None)

        # fprs, tprs, thresholds = roc_curve(data_test[:, 0], predict_proba)
        # right_rate_min_score = right_rate_min(data_test[:, 0], predict, choose_lable(data_train))
        # plt.plot(fprs, tprs, 'slategray')

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'f1': f1,
            'g_mean': g_mean,
            'ap':ap
        }

    # multi class
    else:
        clf.fit(data_train[:, 1:], data_train[:, 0])
        predict = clf.predict(data_test[:, 1:])
        # predict_proba = clf.predict_proba(data_test[:, 1:])[:, 1] 
        predict_proba = clf.predict_proba(data_test[:, 1:])

        accuracy = accuracy_score(data_test[:, 0], predict)
        precision = precision_score(data_test[:, 0], predict, average='macro')
        recall = recall_score(data_test[:, 0], predict, average='macro')
        auc = roc_auc_score(data_test[:, 0], predict_proba, multi_class='ovo',average='macro')  
        f1 = f1_score(data_test[:, 0], predict, average='macro')

        # 计算macro_gmean
        p_class, r_class, f_class, support_micro = precision_recall_fscore_support(
            y_true=data_test[:,0], y_pred=predict, beta=0.5, average=None, warn_for=("precision", "recall", "f-score"),
        )
        macro_gmean = 0
        for i in range(len(p_class)):
            macro_gmean += math.sqrt(p_class[i]*r_class[i])/len(p_class)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'f1': f1,
            'g_mean': macro_gmean,
        }

    return metrics


def get_classifier(classifier, random_state=0):

    if classifier == 'BPNN':
        clf = MLPClassifier(random_state=random_state)
    elif classifier == 'KNN':
        clf = KNeighborsClassifier()
    elif classifier == 'SVM':
        # kernel='rbf', gamma='auto',
        clf = SVC(random_state=random_state, probability=True)
    elif classifier == 'DTree':
        clf = DecisionTreeClassifier(random_state=random_state)
    elif classifier == 'LR':
        clf = LogisticRegression(
            random_state=random_state)  # solver='liblinear',
    elif classifier == 'RF':
        clf = RandomForestClassifier(random_state=random_state)
    elif classifier == 'GBDT':
        clf = GradientBoostingClassifier(random_state=random_state)
    elif classifier == 'AdaBoost':
        clf = AdaBoostClassifier(random_state=random_state)
    elif classifier == 'XGBoost':
        clf = XGBClassifier(random_state=random_state)
    elif classifier == 'LightGBM':
        clf = LGBMClassifier(random_state=random_state)
    else:
        assert 0 == 1, '{} is not exist!'.format(classifier)
    return clf


def write_excel(path, value, table_head=None, sheet_name='sheet1', blank_space=False):
    """
Write a table. If the table does not exist, create a new table. If the table exists, add a new row at the end.
:param path: save path (ending with .xls)
:param value: value, two-dimensional list
:param table_head: first row (table header)
:param sheet_name:
:param blank_space:
:return:
    """
    if not isinstance(value, list):
        value_new = []
        for line in value:
            value_new.append(list(line))
        value = value_new
    try:  
        value_write = value
        if blank_space and value:
            value2 = [['' for _ in range(len(value[0]))]]
            value2.extend(value[1:])
            value_write = value2

        # print(path)
        workbook = xlrd.open_workbook(path)  
        sheets = workbook.sheet_names()  
        worksheet = workbook.sheet_by_name(sheets[0])  
        if table_head and worksheet.row_values(0)[:min(len(table_head), len(worksheet.row_values(0)))] != table_head:
            value2 = [table_head]
            value2.extend(value_write[1:])
            value_write = value2
        rows_old = worksheet.nrows  
        new_workbook = copy(workbook)  
        new_worksheet = new_workbook.get_sheet(0)  

        index = len(value_write)  
        for i in range(0, index):
            for j in range(0, len(value_write[i])):
                new_worksheet.write(i + rows_old, j, value_write[i][j])
        new_workbook.save(path)  

    except FileNotFoundError:  
        value_write = [table_head] if table_head else []
        value_write.extend(value[1:])

        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet(sheet_name)

        index = len(value_write)
        for i in range(0, index):
            for j in range(0, len(value_write[i])):
                sheet.write(i, j, value_write[i][j])
        workbook.save(path)


def choose_lable(data):
    tmp = 0
    label = 0
    for key, value in Counter(data[:, 0]).items():
        if value > tmp:
            label = key
            tmp = value
    return label


def right_rate_min(alist, blist, lable):  
    n = len(set(alist))
    cata_list = [[] for _ in range(n - 1)]  
    right_list = [[] for _ in range(n - 1)]  
    index_list = 0
    sum = 0
    for i in range(len(alist)):
        flag = 1
        if alist[i] == lable:
            continue
        else:
            if alist[i] == blist[i]:
                for j in range(n - 1):
                    if (right_list[j]) and alist[i] == right_list[j][0]:
                        right_list[j].append(alist[i])
                        flag = 0
                        break
                if flag:
                    right_list[index_list].append(alist[i])
                    index_list += 1

    for i in range(len(alist)):
        if alist[i] == lable:
            continue
        else:
            for j in range(n - 1):
                if (right_list[j]) and alist[i] == right_list[j][0]:
                    cata_list[j].append(alist[i])

    for i in range(n - 1):
        if right_list[i] and cata_list[i]:
            sum += (len(right_list[i]) / len(cata_list[i])) / (n - 1)
    return sum


def sampling_methods(data_train=None, method=None, parameters=(30, 5), random_state=42, weight=None, nans=None):
    """
:param data_train: training set
:param method: sampling method
:param parameters: parameters (omit if none)
:param random_state: can be omitted
:return: return the sampled result
    """
    if method == 'none':
        return data_train

#   Pre-process-------------------------------------------------------------------------

    # original
    if method == 'smote':
        model = over_sampling.SMOTE(random_state=random_state,)
    elif method == 'borderline1-smote':
        model = over_sampling.BorderlineSMOTE(
            kind='borderline-1', random_state=random_state,)
    elif method == 'kmeans-smote':
        # cluster_balance_threshold 
        model = over_sampling.KMeansSMOTE(cluster_balance_threshold = 0.03,random_state=random_state)
    elif method == 'ADASYN':
        model = _smote_variants_v1_orignal.ADASYN(random_state=random_state)
    # elif method =='svm-smote':
    #     model = over_sampling.SVMSMOTE(random_state=random_state)
    elif method == 'MWMOTE':
        model = _smote_variants_v1_orignal.MWMOTE(random_state=random_state,)

    # weighted
    elif method == 'smote-Nan':
        model = _imblearn_v5_NanRd_v2.SMOTE(
            random_state=random_state, weight=weight, nans=nans)
    elif method == 'boderline1-smote-Nan':
        model = _imblearn_v5_NanRd_v2.BorderlineSMOTE(
            random_state=random_state, weight=weight, nans=nans)
    elif method == 'kmeans-smote-Nan':
        model = _imblearn_v5_NanRd_v2.KMeansSMOTE(
            random_state=random_state, weight=weight, nans=nans)
    # elif method == 'ADASYN-weight':
    #     model = _smote_variants_v3.ADASYN(random_state=random_state)
    # elif method =='weight-svm-smote':
    #     model = all_smote_v3.SVMSMOTE(random_state=random_state,weight=weight,ntree=ntree)
# ------------------------------------------------------------------------


#   Post-process------------------------------------------------------------------------

    # original
    elif method == 'SMOTE_ENN':
        model = _smote_variants_v1_orignal.SMOTE_ENN(random_state=random_state,)
    elif method == 'SMOTE_TomekLinks':
        model = _smote_variants_v1_orignal.SMOTE_TomekLinks(random_state=random_state,)
    elif method == 'SMOTE_RSB':
        model = _smote_variants_v1_orignal.SMOTE_RSB(random_state=random_state,)
    elif method == 'SMOTE_IPF':
        model = _smote_variants_v1_orignal.SMOTE_IPF(random_state=random_state,)

    elif method == 'MSYN_':
        model = _smote_variants_v1_orignal.MSYN(random_state=random_state)
    elif method == 'SVM_balance':
        model = _smote_variants_v1_orignal.SVM_balance(random_state=random_state)
    elif method == 'ISOMAP_Hybrid':
        model = _smote_variants_v1_orignal.ISOMAP_Hybrid(random_state=random_state)
    elif method == 'SMOTE_FRST_2T':
        model = _smote_variants_v1_orignal.SMOTE_FRST_2T(random_state=random_state)
    elif method == 'DSRBF_':
        model = _smote_variants_v1_orignal.DSRBF(random_state=random_state)
    elif method == 'SMOTE_PSOBAT':
        model = _smote_variants_v1_orignal.SMOTE_PSOBAT(random_state=random_state)

    # weighted
    elif method == 'SMOTE_ENN-Nan':
        model = _smote_variants_v5_NanRd_v2.SMOTE_ENN(random_state=random_state)
    elif method == 'SMOTE_TomekLinks-Nan':
        model = _smote_variants_v5_NanRd_v2.SMOTE_TomekLinks(random_state=random_state)
    elif method == 'SMOTE_RSB-Nan':
        model = _smote_variants_v5_NanRd_v2.SMOTE_RSB(random_state=random_state)
    elif method == 'SMOTE_IPF-Nan':
        model = _smote_variants_v5_NanRd_v2.SMOTE_IPF(random_state=random_state)

    elif method == 'MSYN_-Nan':
        model = _smote_variants_v5_NanRd_v2.MSYN(random_state=random_state)
    elif method == 'SVM_balance-Nan':
        model = _smote_variants_v5_NanRd_v2.SVM_balance(random_state=random_state)
    elif method == 'ISOMAP_Hybrid-Nan':
        model = _smote_variants_v5_NanRd_v2.ISOMAP_Hybrid(random_state=random_state)
    elif method == 'SMOTE_FRST_2T-Nan':
        model = _smote_variants_v5_NanRd_v2.SMOTE_FRST_2T(random_state=random_state)
    elif method == 'DSRBF_-Nan':
        model = _smote_variants_v5_NanRd_v2.DSRBF(random_state=random_state)
    elif method == 'SMOTE_PSOBAT-Nan':
        model = _smote_variants_v5_NanRd_v2.SMOTE_PSOBAT(random_state=random_state)
# ------------------------------------------------------------------------------------


# Others --------------------------------------------------------------------------------
    elif method == 'wkb':
        model =  _smote_variants_v7_wkb.KMeansSMOTE(random_state=42, kind='kmeans-borderline')
    elif method == 'Gaussian_SMOTE':
        model = _smote_variants_v1_orignal.Gaussian_SMOTE(random_state=random_state)
    elif method == 'DBSMOTE':
        model = _smote_variants_v1_orignal.DBSMOTE(random_state=random_state,)
    elif method == 'MOT2LD_':
        model = _smote_variants_v1_orignal.MOT2LD(random_state=random_state)
    elif method == 'SMOTE-GB':
        model = _smote_variants_gbsmote.SMOTE(random_state=random_state)

    elif method == 'RSMOTE':

        # model = RSmote(ir=1, k=5, data=data_train)
        # return model.over_sampling()
        
        model = RSmoteKClasses(ir=1, k=5,)
        X_resampled, y_resampled = model.fit_resample(data_train[:, 1:], data_train[:, 0])
        y_resampled = y_resampled.reshape(y_resampled.shape[0], 1)
        data_train_resampled = np.hstack((y_resampled, X_resampled))
        return data_train_resampled

    elif method == 'AdaptiveSMOTE': 
        model = ASmote(data=data_train)
        X, Xn, Yn, Xp, Yp = model.dataReset()
        Xp2 = oversampleASmote(X, Xn, Xp)
        Y = np.hstack((Yn*np.ones(len(Xn)), Yp*np.ones(len(Xp2)))
                      ).reshape(-1, 1)
        data_t = np.vstack((Xn, Xp2))
        data_r = np.hstack((Y, data_t))
        return data_r

    elif method == 'LDAS':
        """Ldas"""
        X_resampled, y_resampled = LDAS().fit_sample(
            data_train[:, 1:], data_train[:, 0])
        y_resampled = y_resampled.reshape(y_resampled.shape[0], 1)
        data_train_resampled = np.hstack((y_resampled, X_resampled))
        return data_train_resampled

    # elif method == 'Geometric SMOTE':
    #     geometric_smote = GeometricSMOTE()
    #     X_resampled, y_resampled = geometric_smote.fit_resample(
    #         data_train[:, 1:], data_train[:, 0])
    #     y_resampled = y_resampled.reshape(y_resampled.shape[0], 1)
    #     data_train_resampled = np.hstack((y_resampled, X_resampled))
    #     return data_train_resampled

    elif method == 'GDO':
        """GDO_TKDE"""
        X_resampled, y_resampled = GDO(k=5, alpha=1).fit_sample(
            data_train[:, 1:], data_train[:, 0])
        y_resampled = y_resampled.reshape(y_resampled.shape[0], 1)
        data_train_resampled = np.hstack((y_resampled, X_resampled))
        return data_train_resampled

    elif method == 'GBSY':
        X_resampled, y_resampled = main(data_train[:, 1:], data_train[:, 0],1)
        # print(type(X_resampled), X_resampled.shape,type(y_resampled),len(y_resampled))
        y_resampled = np.asarray(y_resampled)
        # print(type(X_resampled), X_resampled.shape,type(y_resampled),len(y_resampled))
        y_resampled = y_resampled.reshape(y_resampled.shape[0], 1)
        data_train_resampled = np.hstack((y_resampled, X_resampled))
        return data_train_resampled
        

# --------------------------------------------------------------------------------

    else:
        raise Exception(
            r'the sampling method name \'{}\' is not exist!'.format(method))

    if '_' in method:  # smote_variants
        if 'Nan' not in method:      # origin
            X_resampled, y_resampled = model.sample(
                data_train[:, 1:], data_train[:, 0])
        else:       # weight
            X_resampled, y_resampled = model.sample(
                data_train[:, 1:], data_train[:, 0], weight, nans)
    else:   # imbalanced_learn
        X_resampled, y_resampled = model.fit_resample(
            data_train[:, 1:], data_train[:, 0])

    y_resampled = y_resampled.reshape(y_resampled.shape[0], 1)
    data_train_resampled = np.hstack((y_resampled, X_resampled))

    return data_train_resampled 


def sampling_methods_General(data_train=None, method=None, parameters=(30, 5), random_state=42,k_neighbors=5):
    """
:param data_train: training set
:param method: sampling method
:param parameters: parameters (omit if none)
:param random_state: can be omitted
:return: return the sampled result
    """
    if method == 'none':
        return data_train

#   Pre-process-------------------------------------------------------------------------
    if method == 'smote':
        model = over_sampling.SMOTE(random_state=random_state,k_neighbors=k_neighbors)
    elif method == 'borderline1-smote':
        model = over_sampling.BorderlineSMOTE(
            kind='borderline-1', random_state=random_state,k_neighbors=k_neighbors)
    elif method == 'kmeans-smote':
        # cluster_balance_threshold 
        model = over_sampling.KMeansSMOTE(cluster_balance_threshold = 0.03,random_state=random_state,k_neighbors=k_neighbors)
    elif method == 'ADASYN':
        model = _smote_variants_v1_orignal.ADASYN(random_state=random_state,n_neighbors=k_neighbors)
    # elif method =='svm-smote':
    #     model = over_sampling.SVMSMOTE(random_state=random_state)
    elif method == 'MWMOTE':
        model = _smote_variants_v1_orignal.MWMOTE(random_state=random_state,k1=k_neighbors,k2=k_neighbors,k3=k_neighbors)

# ------------------------------------------------------------------------


#   Post-process------------------------------------------------------------------------

    elif method == 'SMOTE_ENN':
        model = _smote_variants_v1_orignal.SMOTE_ENN(random_state=random_state,n_neighbors=k_neighbors)
    elif method == 'SMOTE_TomekLinks':
        model = _smote_variants_v1_orignal.SMOTE_TomekLinks(random_state=random_state,n_neighbors=k_neighbors)
    elif method == 'SMOTE_RSB':
        model = _smote_variants_v1_orignal.SMOTE_RSB(random_state=random_state,n_neighbors=k_neighbors)
    elif method == 'SMOTE_IPF':
        model = _smote_variants_v1_orignal.SMOTE_IPF(random_state=random_state,n_neighbors=k_neighbors)

    elif method == 'MSYN_':
        model = _smote_variants_v1_orignal.MSYN(random_state=random_state)
    elif method == 'SVM_balance':
        model = _smote_variants_v1_orignal.SVM_balance(random_state=random_state)
    elif method == 'ISOMAP_Hybrid':
        model = _smote_variants_v1_orignal.ISOMAP_Hybrid(random_state=random_state)
    elif method == 'SMOTE_FRST_2T':
        model = _smote_variants_v1_orignal.SMOTE_FRST_2T(random_state=random_state)
    elif method == 'DSRBF_':
        model = _smote_variants_v1_orignal.DSRBF(random_state=random_state)
    elif method == 'SMOTE_PSOBAT':
        model = _smote_variants_v1_orignal.SMOTE_PSOBAT(random_state=random_state)

# ------------------------------------------------------------------------------------


# Others --------------------------------------------------------------------------------
    elif method == 'wkb':
        model =  _smote_variants_v7_wkb.KMeansSMOTE(random_state=42, kind='kmeans-borderline',k_neighbors=k_neighbors)
    # elif method == 'Gaussian_SMOTE':
        # model = _smote_variants_v1.Gaussian_SMOTE(random_state=random_state,k_neighbors=k_neighbors)
    elif method == 'DBSMOTE':
        model = _smote_variants_v1_orignal.DBSMOTE(random_state=random_state,min_samples=k_neighbors)
    elif method == 'MOT2LD_':
        model = _smote_variants_v1_orignal.MOT2LD(random_state=random_state)

    elif method == 'RSMOTE':
        
        # model = RSmote(ir=1, k=5, data=data_train)
        # return model.over_sampling()
        
        model = RSmoteKClasses(ir=1, k=k_neighbors,)
        X_resampled, y_resampled = model.fit_resample(data_train[:, 1:], data_train[:, 0])
        y_resampled = y_resampled.reshape(y_resampled.shape[0], 1)
        data_train_resampled = np.hstack((y_resampled, X_resampled))
        return data_train_resampled

    elif method == 'AdaptiveSMOTE':
        model = ASmote(data=data_train)
        X, Xn, Yn, Xp, Yp = model.dataReset()
        Xp2 = oversampleASmote(X, Xn, Xp)
        Y = np.hstack((Yn*np.ones(len(Xn)), Yp*np.ones(len(Xp2)))
                      ).reshape(-1, 1)
        data_t = np.vstack((Xn, Xp2))
        data_r = np.hstack((Y, data_t))
        return data_r

    elif method == 'LDAS':
        """Ldas"""
        X_resampled, y_resampled = LDAS(k=k_neighbors, k1=k_neighbors,).fit_sample(
            data_train[:, 1:], data_train[:, 0])
        y_resampled = y_resampled.reshape(y_resampled.shape[0], 1)
        data_train_resampled = np.hstack((y_resampled, X_resampled))
        return data_train_resampled

    # elif method == 'Geometric SMOTE':
    #     geometric_smote = GeometricSMOTE()
    #     X_resampled, y_resampled = geometric_smote.fit_resample(
    #         data_train[:, 1:], data_train[:, 0])
    #     y_resampled = y_resampled.reshape(y_resampled.shape[0], 1)
    #     data_train_resampled = np.hstack((y_resampled, X_resampled))
    #     return data_train_resampled

    elif method == 'GDO':
        """GDO_TKDE"""
        X_resampled, y_resampled = GDO(k=k_neighbors, alpha=1).fit_sample(
            data_train[:, 1:], data_train[:, 0])
        y_resampled = y_resampled.reshape(y_resampled.shape[0], 1)
        data_train_resampled = np.hstack((y_resampled, X_resampled))
        return data_train_resampled

    elif method == 'GBSY':
        X_resampled, y_resampled = main(data_train[:, 1:], data_train[:, 0],1)
        # print(type(X_resampled), X_resampled.shape,type(y_resampled),len(y_resampled))
        y_resampled = np.asarray(y_resampled)
        # print(type(X_resampled), X_resampled.shape,type(y_resampled),len(y_resampled))
        y_resampled = y_resampled.reshape(y_resampled.shape[0], 1)
        data_train_resampled = np.hstack((y_resampled, X_resampled))
        return data_train_resampled
    elif method == 'MC_CCR':
        pass
    elif method == 'MC_RBO':
        pass
    

# --------------------------------------------------------------------------------

    else:
        raise Exception(
            r'the sampling method name \'{}\' is not exist!'.format(method))

    if '_' in method:  # smote_variants
        if 'Nan' not in method:      # origin
            X_resampled, y_resampled = model.sample(
                data_train[:, 1:], data_train[:, 0])
        else:       # weight
            X_resampled, y_resampled = model.sample(
                data_train[:, 1:], data_train[:, 0], weight, nans)
    else:   # imbalanced_learn
        X_resampled, y_resampled = model.fit_resample(
            data_train[:, 1:], data_train[:, 0])

    y_resampled = y_resampled.reshape(y_resampled.shape[0], 1)
    data_train_resampled = np.hstack((y_resampled, X_resampled))

    return data_train_resampled


def __test_sampling():
    data_names = ['anneal', 'credit', 'german', 'heart1', 'heart2', 'hepatitis', 'horse', 'iono', 'sonar', 'wdbc',
                  'wine', 'mushroom', 'soybean']

    methods = ('smote', 'borderline1_smote', 'borderline2_smote',
               'svm_smote', 'adasyn', 'kmeans_smote')

    for data_name in data_names:

        data, counter = load_k_classes_data(data_name)

        print('data set: {:15s}  samples: {:6d}   features: {:3d}   classes:{:2d}   counter:{}'
              .format(data_name, data.shape[0], data.shape[1], len(counter), counter))

        print('{:20s}：'.format(''), [counter[k]
                                        for k in range(len(counter))])

        for method in methods:
            data_train_resampled = sampling_methods(data, method)
            counter2 = Counter(data_train_resampled[:, 0])

            print('{:20s}：'.format(method), [
                  counter2[k] for k in range(len(counter2))])


def __test_fourclass_plot():
    data_name = 'fourclass'
    noise_rate = 0.0
    data_train, data_test = load_k_classes_data(
        data_name, noise_rate=noise_rate)
    plot_data_and_balls(data_train, data_train)


def __test_excel():
    data = load_2classes_data('svmguide1')[0]
    print(data)
    # write_excel('abc.xls', data)


if __name__ == '__main__':
    __test_excel()
    __test_fourclass_plot()
    __test_sampling()
