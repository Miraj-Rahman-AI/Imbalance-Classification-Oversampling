# Import necessary libraries
from experiment_v5_NanRd import (load_2classes_data, load_k_classes_data_lm, get_classifier, get_metrics,
                              write_excel, now_time_str, sampling_methods, mean_std, load_k_classes_data)
import json
from NaN import NaN_RD
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings to keep the output clean

# Function to perform resampling on the training data
def resamplings(data_train, methods):
    """
    Perform resampling on the training data using specified methods.
    
    Parameters:
    - data_train: The training data to be resampled.
    - methods: List of resampling methods to apply.
    
    Returns:
    - data_train_resampleds: Dictionary containing resampled datasets for each method.
    """
    data_train_resampleds = {}
    NData, weight, nans = NaN_RD(data_train)  # Compute relative density and natural neighbors

    for method in methods:
        data_train_resampleds['data_train_' + method] = []
        data_train_resampled = sampling_methods(data_train=data_train,  # Apply resampling method
                                                method=method,
                                                weight=weight,
                                                nans=nans)
        print('Oversampled:\t\t', method)
        data_train_resampleds['data_train_' + method].append(data_train_resampled)
    return data_train_resampleds

# Function to gather information about the datasets
def data_info(data_names: list, noise_rate: int, is_binary: bool):
    """
    Gather information about the datasets, including sample size, features, and class distribution.
    
    Parameters:
    - data_names: List of dataset names.
    - noise_rate: Noise rate to be applied to the datasets.
    - is_binary: Boolean indicating whether the datasets are binary or multi-class.
    """
    uci_extended = ['breast_tissue2', 'ecoli', 'glass', 'haberman', 'heart', 'iris', 'libra', 'liver_disorders',
                    'pima', 'segment', 'vehicle', 'wine', 'simulated1', 'simulated2', 'simulated3', 'simulated4',
                            'simulated5', 'simulated6', 'simulated7', 'simulated8', 'simulated9', 'simulated10']
    imbs = {'splice': 5, 'breastcancer': 5, 'creditApproval': 5, 'votes': 5, 'Online_Retail': 50,
            'codrna': 20, 'nomao': 20, 'mocap': 40, 'poker': 50, 'skin': 100, 'covtype': 60, 'comedy': 100}
    columns = ['data_name', 'samples', 'Features', 'Maj', 'Min','k_class','dicts',]
    lists = []

    for d, data_name in enumerate(data_names):
        imb = imbs[data_name] if data_name in imbs.keys() else (
            0 if data_name in uci_extended else 10)
        if is_binary:
            data_train, data_test, counters = load_2classes_data(
                data_name, noise_rate=noise_rate, imbalance_rate=imb)  # Load binary classification dataset
        else:   # Multi-class
            data_train, data_test, counters = load_k_classes_data(
                data_name, noise_rate=noise_rate)  # Load multi-class dataset

        data = np.vstack((data_train, data_test))
        dicts = Counter(data[:, 0])
        dic = dict(dicts)
        lists.append([data_name, data.shape[0],
                      data.shape[1]-1, list(dic.values())[0], list(dic.values())[1], len(dicts),dicts])

    df = pd.DataFrame(lists, columns=columns, index=None)
    df.to_csv(r'KBsomte\SMOTE\data_infomation/data_info_' + str(is_binary) +
              str(noise_rate) + '.csv', index=False)

# Function to run the standard evaluation pipeline
def run_std(result_file_path_mean, result_file_path_std, data_names, classifiers, methods, noise_rate, is_binary):
    """
    Run the standard evaluation pipeline, including resampling, cross-validation, and performance evaluation.
    
    Parameters:
    - result_file_path_mean: Path to save the mean results.
    - result_file_path_std: Path to save the standard deviation results.
    - data_names: List of dataset names.
    - classifiers: List of classifiers to evaluate.
    - methods: List of resampling methods to apply.
    - noise_rate: Noise rate to be applied to the datasets.
    - is_binary: Boolean indicating whether the datasets are binary or multi-class.
    """
    table_head = ['data', 'Classifier', 'sampling method', 'noise rate', 'samples', 'features', 'train 0',
                  'train 1', 'test 0', 'test 1', 'sampled 0', 'sampled 1', 'accuracy', 'precision', 'recall', 'AUC',
                  'f1', 'g_mean', 'right_rate_min', 'tp', 'fp', 'fn', 'tn', 'now_time']

    uci_extended = ['breast_tissue2', 'ecoli', 'glass', 'haberman', 'heart', 'iris', 'libra', 'liver_disorders',
                    'pima', 'segment', 'vehicle', 'wine', 'simulated1', 'simulated2', 'simulated3', 'simulated4',
                            'simulated5', 'simulated6', 'simulated7', 'simulated8', 'simulated9', 'simulated10']
    imbs = {'splice': 5, 'breastcancer': 5, 'creditApproval': 5, 'votes': 5, 'Online_Retail': 50,
            'codrna': 20, 'nomao': 20, 'mocap': 40, 'poker': 50, 'skin': 100, 'covtype': 60, 'comedy': 100}

    debug = []
    for d, data_name in enumerate(data_names):
        imb = imbs[data_name] if data_name in imbs.keys() else (
            0 if data_name in uci_extended else 10)
        if is_binary:   # Binary classification
            data_train, data_test, counters = load_2classes_data(
                data_name, noise_rate=noise_rate, imbalance_rate=imb)  # Load binary classification dataset
        else:   # Multi-class
            data_train, data_test, counters = load_k_classes_data(
                data_name, noise_rate=noise_rate)  # Load multi-class dataset

        data_train = np.vstack([data_train, data_test])

        print('\n\nnoise rate: {:.2f}  data set {:2d}/{:2d}: {:15s}  samples: {:6d}   features: {:3d}   '
              'train 0: {:6d}   train 1: {:6d}   test 0: {:6d}   test 1: {:6d}\n'
              .format(noise_rate, d + 1, len(data_names), data_name,
                      counters[0], counters[1], counters[2], counters[3], counters[4], counters[5]))

        try:
            data_train_resampleds = resamplings(data_train, methods=methods)
        except Exception as e:
            debug.append([noise_rate, data_name, str(e)])
            continue

        for classifier in classifiers:
            print('    classifier: {:10s}'.format(classifier))
            clf = get_classifier(classifier)
            for method in methods:
                try:
                    data_train_those = data_train_resampleds['data_train_' + method]
                except Exception as e:
                    debug.append([noise_rate, data_name, method, classifier, str(e)])
                    continue

                metri = []
                kf = StratifiedShuffleSplit(n_splits=10)

                try:
                    for train_index, validate_index in kf.split(data_train_those[0][:, 1:], data_train_those[0][:, 0]):
                        train, validate = data_train_those[0][train_index], data_train_those[0][validate_index]
                        metrics = get_metrics(clf, train, validate)  # Get evaluation metrics
                        metri.append(metrics)
                    
                    metric_std = {'accuracy': [], 'precision': [], 'recall': [], 'auc': [], 'f1': [], 'g_mean': []}
                    for i in range(0, len(metri)):  # Aggregate results from 10-fold cross-validation
                        for k, v in metri[i].items():
                            if k in metric_std:
                                metric_std[k].append(v)

                    for k, v in metric_std.items():  # Calculate mean and standard deviation
                        metri[0][k] = np.mean(v)
                        metri[1][k] = np.std(v)

                    metrics = metri[0]  # Mean metrics
                    excel_line = [data_name, classifier, method, noise_rate,
                                counters[0], counters[1], counters[2], counters[3], counters[4], counters[5],
                                Counter(train[:, 0])[0], Counter(train[:, 0])[1],
                                metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['auc'],
                                metrics['f1'], metrics['g_mean'], now_time_str()]
                    write_excel(result_file_path_mean, [], table_head=table_head)
                    print('        {}  {:20s}'.format(now_time_str(), method), metrics)
                    write_excel(result_file_path_mean, [excel_line], table_head=table_head)

                    metrics = metri[1]  # Standard deviation metrics
                    excel_line = [data_name, classifier, method, noise_rate,
                                counters[0], counters[1], counters[2], counters[3], counters[4], counters[5],
                                Counter(train[:, 0])[0], Counter(train[:, 0])[1],
                                metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['auc'],
                                metrics['f1'], metrics['g_mean'], now_time_str()]
                    write_excel(result_file_path_std, [], table_head=table_head)
                    print('        {}  {:20s}'.format(now_time_str(), method), metrics)
                    write_excel(result_file_path_std, [excel_line], table_head=table_head)
                except Exception as e:
                    continue

    # Save debug information to a CSV file
    df = pd.DataFrame(debug)
    df.to_csv(r'KBsomte\debug'+str(noise_rate)+'.csv', index=False)

# Main function to execute the pipeline
def main(is_binary: bool):
    """
    Main function to execute the pipeline for binary or multi-class classification.
    
    Parameters:
    - is_binary: Boolean indicating whether to run binary or multi-class classification.
    """
    if is_binary:
        data_names = [
            'default of credit card clients', 'nomao', 'susy', 
        ]
    else:
        data_names = [
            'frogs',
        ]

    classifiers = [
        'LR', 'KNN', 'BPNN', 'DTree', 'AdaBoost', 'GBDT',
    ]

    methods = (
        'kmeans-smote',   # Resampling method
    )

    result_file_path_mean = r'KBsomte/SMOTE/result table/{}_{} result Gsmote.xls'.format('mean', now_time_str(colon=False))
    result_file_path_std = r'KBsomte/SMOTE/result table/{}_{} result Gsmote.xls'.format('std', now_time_str(colon=False))

    run_std(result_file_path_mean, result_file_path_std, data_names, classifiers, methods, noise_rate=0, is_binary=is_binary)
    run_std(result_file_path_mean, result_file_path_std, data_names, classifiers, methods, noise_rate=0.1, is_binary=is_binary)
    run_std(result_file_path_mean, result_file_path_std, data_names, classifiers, methods, noise_rate=0.2, is_binary=is_binary)
    run_std(result_file_path_mean, result_file_path_std, data_names, classifiers, methods, noise_rate=0.3, is_binary=is_binary)

if __name__ == '__main__':
    main(is_binary=0)    # Run multi-class classification
    # main(is_binary=1)  # Run binary classification