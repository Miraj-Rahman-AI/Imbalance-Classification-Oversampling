# Import necessary libraries
import math
import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
import json
import timeout_decorator
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings to keep the output clean

# Custom module imports
from RSDS import RSDS_zhou  # Import RSDS_zhou for data denoising and weighting
from experiments_all import (  # Import custom functions for data loading, model evaluation, etc.
    load_2classes_data, load_k_classes_data, load_kto2_classes_data_lm, get_classifier, get_metrics,
    write_excel, now_time_str, sampling_methods, mean_std
)
from NaN import NaN_RD  # Import NaN_RD for handling missing data

# Function to perform resampling on the training data with a timeout decorator
@timeout_decorator.timeout(1800)  # Set a timeout of 1800 seconds (30 minutes) for this function
def resamplings(data_train, methods):
    """
    Perform resampling on the training data using specified methods.
    
    Parameters:
    - data_train: The training dataset (numpy array).
    - methods: List of resampling methods to apply.
    
    Returns:
    - data_train_resampleds: Dictionary containing resampled datasets for each method.
    """
    data_train_resampleds = {}
    
    # Calculate the number of trees for RSDS_zhou based on the dataset size
    ntree = int(math.log(data_train.shape[0] * data_train.shape[1] - 1, 2) / 2
    
    # Apply RSDS_zhou to denoise and weight the dataset
    RData, weight_SW = RSDS_zhou(data_train, ntree)  # RData: denoised dataset, weight_SW: weight matrix
    
    # Apply NaN_RD to handle missing data
    NData, weight_WNND, nans = NaN_RD(data_train)  # NData: original data, weight_WNND: relative density, nans: natural neighbors

    # Loop through each resampling method
    for method in methods:
        data_train_resampleds['data_train_' + method] = []
        
        # Apply the resampling method to the training data
        data_train_resampled = sampling_methods(
            data_train=data_train,  # Use the denoised and weighted data
            method=method,
            weight_SW=weight_SW,
            ntree=ntree,
            RData=RData,
            weight_WNND=weight_WNND,
            nans=nans,
        )
        print('Oversampled:\t\t', method)
        
        # Store the resampled data in the dictionary
        data_train_resampleds['data_train_' + method].append(data_train_resampled)
    
    return data_train_resampleds


# Function to gather information about the datasets
def data_info(data_names: list, noise_rate: int):
    """
    Gather information about the datasets, such as the number of samples, features, and class distribution.
    
    Parameters:
    - data_names: List of dataset names.
    - noise_rate: The noise rate applied to the datasets.
    
    Returns:
    - A CSV file containing the dataset information.
    """
    # Define extended UCI datasets and imbalanced datasets
    uci_extended = ['breast_tissue2', 'ecoli', 'glass', 'haberman', 'heart', 'iris', 'libra', 'liver_disorders',
                    'pima', 'segment', 'vehicle', 'wine', 'simulated1', 'simulated2', 'simulated3', 'simulated4',
                    'simulated5', 'simulated6', 'simulated7', 'simulated8', 'simulated9', 'simulated10']
    imbs = {'splice': 5, 'breastcancer': 5, 'creditApproval': 5, 'votes': 5, 'Online_Retail': 50,
            'codrna': 20, 'nomao': 20, 'mocap': 40, 'poker': 50, 'skin': 100, 'covtype': 60, 'comedy': 100}
    
    # Define columns for the output CSV
    columns = ['data_name', 'samples', 'Features', 'Maj', 'Min']
    lists = []

    # Loop through each dataset and gather information
    for d, data_name in enumerate(data_names):
        # Determine the imbalance rate for the dataset
        imb = imbs[data_name] if data_name in imbs.keys() else (
            0 if data_name in uci_extended else 10)
        
        # Load the dataset with the specified noise and imbalance rates
        data_train, data_test, counters = load_2classes_data(
            data_name, noise_rate=noise_rate, imbalance_rate=imb)
        
        # Combine training and test data
        data = np.vstack((data_train, data_test))
        
        # Count the number of samples in each class
        dicts = Counter(data[:, 0])
        
        # Append the dataset information to the list
        lists.append([data_name, data.shape[0], data.shape[1] - 1, dicts[0.0], dicts[1.0]])

    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(lists, columns=columns, index=None)
    df.to_csv(r'KBsomte\SMOTE\data_infomation/data_info_' + str(noise_rate) + '.csv', index=False)


# Function to run the experiments
def run(data_names, classifiers, methods, noise_rate, is_binary: bool, is_raise: bool, result_file_path_mean=None, result_file_path_std=None):
    """
    Run the experiments on the specified datasets using the given classifiers and resampling methods.
    
    Parameters:
    - data_names: List of dataset names.
    - classifiers: List of classifiers to use.
    - methods: List of resampling methods to apply.
    - noise_rate: The noise rate applied to the datasets.
    - is_binary: Whether the dataset is binary (1) or multi-class (0).
    - is_raise: Whether to raise exceptions or log them.
    - result_file_path_mean: Path to save the mean results.
    - result_file_path_std: Path to save the standard deviation results.
    """
    # Define the table headers for the results
    table_head = ['data', 'Classifier', 'sampling method', 'noise rate', 'samples', 'features', 'train 0',
                  'train 1', 'test 0', 'test 1', 'sampled 0', 'sampled 1', 'accuracy', 'precision', 'recall', 'AUC',
                  'f1', 'g_mean', 'right_rate_min', 'tp', 'fp', 'fn', 'tn', 'now_time']

    # Define extended UCI datasets and imbalanced datasets
    uci_extended = ['breast_tissue2', 'ecoli', 'glass', 'haberman', 'heart', 'iris', 'libra', 'liver_disorders',
                    'pima', 'segment', 'vehicle', 'wine', 'simulated1', 'simulated2', 'simulated3', 'simulated4',
                    'simulated5', 'simulated6', 'simulated7', 'simulated8', 'simulated9', 'simulated10']
    imbs = {'splice': 5, 'breastcancer': 5, 'creditApproval': 5, 'votes': 5, 'Online_Retail': 50,
            'codrna': 20, 'nomao': 20, 'mocap': 40, 'poker': 50, 'skin': 100, 'covtype': 60, 'comedy': 100}

    debug = []  # List to store debug information

    # Loop through each dataset
    for d, data_name in enumerate(data_names):
        # Determine the imbalance rate for the dataset
        imb = imbs[data_name] if data_name in imbs.keys() else (
            0 if data_name in uci_extended else 10)
        
        # Load the dataset based on whether it is binary or multi-class
        if is_binary == 1:  # Binary datasets
            data_train, data_test, counters = load_2classes_data(
                data_name, noise_rate=noise_rate, imbalance_rate=imb)
        elif is_binary == 2:  # Multiclass to binary
            data_train, data_test, counters = load_kto2_classes_data_lm(
                data_name, noise_rate=noise_rate)
        elif is_binary == 0:  # Multiclass datasets
            data_train, data_test, counters = load_k_classes_data(
                data_name, noise_rate=noise_rate)
        
        # Combine training and test data
        data_train = np.vstack([data_train, data_test])

        # Print dataset information
        print('\n\nnoise rate: {:.2f}  data set {:2d}/{:2d}: {:15s}  samples: {:6d}   features: {:3d}   '
              'train 0: {:6d}   train 1: {:6d}   test 0: {:6d}   test 1: {:6d}\n'
              .format(noise_rate, d + 1, len(data_names), data_name,
                      counters[0], counters[1], counters[2], counters[3], counters[4], counters[5]))

        # Apply resampling methods to the training data
        try:
            data_train_resampleds = resamplings(data_train, methods=methods)
        except Exception as e:
            if is_raise:
                raise e
            else:
                debug.append([noise_rate, data_name, str(e)])

        # Loop through each classifier
        for classifier in classifiers:
            print('    classifier: {:10s}'.format(classifier))
            clf = get_classifier(classifier)  # Get the classifier object
            
            # Loop through each resampling method
            for method in methods:
                try:
                    # Extract the resampled dataset
                    data_train_those = data_train_resampleds['data_train_' + method]
                except Exception as e:
                    if is_raise:
                        raise e
                    else:
                        debug.append([noise_rate, data_name, method, classifier, str(e)])

                # Perform 10-fold cross-validation
                metric_skf = {'accuracy': [], 'precision': [], 'recall': [], 'auc': [], 'f1': [], 'g_mean': []}
                skf = StratifiedShuffleSplit(n_splits=10)

                try:
                    for train_index, validate_index in skf.split(data_train_those[0][:, 1:], data_train_those[0][:, 0]):
                        train, validate = data_train_those[0][train_index], data_train_those[0][validate_index]
                        metrics = get_metrics(clf, train, validate)  # Get evaluation metrics
                        for k, v in metrics.items():
                            metric_skf[k].append(v)
                    
                    # Compute mean and standard deviation of the metrics
                    metric_mean = {k: np.mean(v) for k, v in metric_skf.items()}
                    metric_std = {k: np.std(v) for k, v in metric_skf.items()}

                    # Write mean results to CSV
                    if result_file_path_mean:
                        excel_line = [
                            data_name, classifier, method, noise_rate,
                            counters[0], counters[1], counters[2], counters[3], counters[4], counters[5],
                            Counter(train[:, 0])[0], Counter(train[:, 0])[1],
                            metric_mean['accuracy'], metric_mean['precision'], metric_mean['recall'], metrics['auc'],
                            metric_mean['f1'], metric_mean['g_mean'], now_time_str()
                        ]
                        write_excel(result_file_path_mean, [], table_head=table_head)
                        print('{}  {:20s}'.format(now_time_str(), method), metrics)
                        write_excel(result_file_path_mean, [excel_line], table_head=table_head)
                    
                    # Write standard deviation results to CSV
                    if result_file_path_std:
                        excel_line = [
                            data_name, classifier, method, noise_rate,
                            counters[0], counters[1], counters[2], counters[3], counters[4], counters[5],
                            Counter(train[:, 0])[0], Counter(train[:, 0])[1],
                            metric_std['accuracy'], metric_std['precision'], metric_std['recall'], metrics['auc'],
                            metric_std['f1'], metric_std['g_mean'], now_time_str()
                        ]
                        write_excel(result_file_path_std, [], table_head=table_head)
                        print('{}  {:20s}'.format(now_time_str(), method), metrics)
                        write_excel(result_file_path_std, [excel_line], table_head=table_head)
                except Exception as e:
                    if is_raise:
                        raise e
                    else:
                        debug.append([noise_rate, data_name, method, classifier, str(e)])

    # Save debug information to a CSV file
    df = pd.DataFrame(debug)
    df.to_csv(r'KBsomte/SMOTE/Log/' + str(round(noise_rate, 1)) + '.csv', index=False)


# Main function for binary classification
def main_binary(is_binary: bool = 1, is_raise: bool = False):
    """
    Main function to run experiments on binary classification datasets.
    
    Parameters:
    - is_binary: Whether the dataset is binary (1) or multi-class (0).
    - is_raise: Whether to raise exceptions or log them.
    """
    # List of binary classification datasets
    data_names = [
        'new-thyroid1', 'ecoli', 'wisconsin', 'diabetes', 'breastcancer', 'vehicle2', 'vehicle',
        'yeast1', 'Faults', 'segment', 'satimage', 'wilt', 'svmguide1', 'mushrooms', 'page-blocks0',
        'Data_for_UCI_named', 'letter', 'avila', 'magic', 'susy', 'default of credit card clients', 'nomao',
    ]

    # List of classifiers to use
    classifiers = [
        'KNN', 'DTree', 'LR', 'AdaBoost', 'GBDT', 'BPNN',
    ]

    # List of resampling methods to apply
    methods = (
        'SMOTE-IPF', 'SMOTE_IPF-GB',
    )

    # Define the result file path
    result_file_path = r'KBsomte/SMOTE/result table/binary/{} result binary.xls'.format(
        now_time_str(colon=False))

    # Run the experiments with the specified noise rate
    run(result_file_path, data_names, classifiers, methods, noise_rate=0.1, is_binary=is_binary, is_raise=is_raise)


# Main function for multi-class classification
def main_multiClass(is_binary: bool = 0, is_raise: bool = False):
    """
    Main function to run experiments on multi-class classification datasets.
    
    Parameters:
    - is_binary: Whether the dataset is binary (1) or multi-class (0).
    - is_raise: Whether to raise exceptions or log them.
    """
    # List of multi-class classification datasets
    data_names = [
        'soybean', 'contraceptive', 'sensorReadings', 'frogs', 'wine', 'vertebralColumn', 'OBS', 'PhishingData',
        'lymphography', 'heart2', 'zoo', 'ecoli',
    ]

    # List of classifiers to use
    classifiers = [
        'KNN', 'DTree', 'LR', 'AdaBoost', 'GBDT', 'BPNN',
    ]

    # List of resampling methods to apply
    methods = (
        'MC_CCR', 'MC_RBO', 'MDO',
    )

    # Define the result file paths for mean and standard deviation
    result_file_path_mean = r'KBsomte/SMOTE/result table/multiClass/mean/{} mean multiClass.xls'.format(
        now_time_str(colon=False))
    result_file_path_std = r'KBsomte/SMOTE/result table/multiClass/std/{} std multiClass.xls'.format(
        now_time_str(colon=False))

    # Run the experiments with the specified noise rate
    run(data_names, classifiers, methods, noise_rate=0.3, is_binary=is_binary, is_raise=is_raise,
        result_file_path_mean=result_file_path_mean, result_file_path_std=result_file_path_std)


# Entry point of the script
if __name__ == '__main__':
    # main_multiClass(is_binary=0, is_raise=0)  # Run multi-class experiments
    main_binary(is_binary=1, is_raise=0)  # Run binary experiments