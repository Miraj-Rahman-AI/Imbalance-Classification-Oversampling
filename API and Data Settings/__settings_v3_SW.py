# Import necessary libraries
import math  # For mathematical operations
from itertools import count  # For creating iterators
from collections import Counter  # For counting occurrences of elements
import numpy as np  # For numerical computations
import pandas as pd  # For data manipulation and analysis
from sklearn.ensemble import GradientBoostingClassifier  # For using GBDT classifier
from sklearn.datasets import make_classification  # For generating synthetic datasets
from imblearn.over_sampling import SMOTE, ADASYN  # For oversampling techniques
from imblearn.combine import SMOTETomek, SMOTEENN  # For combined sampling techniques
from imblearn.under_sampling import ClusterCentroids  # For undersampling techniques
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit  # For cross-validation
import warnings  # To suppress warnings
warnings.filterwarnings('ignore')  # Ignore warnings to keep the output clean

# Custom module imports
from RSDS import RSDS_zhou  # Import RSDS_zhou for data denoising and weighting
from experiment_v3_SW import (  # Import custom functions for data loading, model evaluation, etc.
    load_2classes_data, load_k_classes_data_lm, get_classifier, get_metrics,
    write_excel, now_time_str, sampling_methods, mean_std
)


# Function to perform resampling on the training data
def resamplings(data_train, methods, ntree):
    """
    Perform resampling on the training data using specified methods.
    
    Parameters:
    - data_train: The training dataset (numpy array).
    - methods: List of resampling methods to apply.
    - ntree: Number of trees for RSDS_zhou.
    
    Returns:
    - data_train_resampleds: Dictionary containing resampled datasets for each method.
    """
    data_train_resampleds = {}  # Dictionary to store resampled datasets
    
    # Calculate the number of trees for RSDS_zhou based on the dataset size
    ntree = int(math.log(data_train.shape[0] * data_train.shape[1], 2) / 2)
    print("ntree**************************************:\t", ntree)
    
    # Apply RSDS_zhou to denoise and weight the dataset
    RData, weight = RSDS_zhou(data_train, ntree)  # RData: denoised dataset, weight: weight matrix

    # Loop through each resampling method
    for method in methods:
        data_train_resampleds['data_train_' + method] = []  # Initialize list for the current method
        
        # Apply the resampling method to the training data
        data_train_resampled = sampling_methods(
            data_train=data_train,  # Original data
            RData=RData,  # Denoised and weighted data
            method=method,
            weight=weight,
            ntree=ntree
        )
        data_train_resampleds['data_train_' + method].append(data_train_resampled)  # Store resampled data
    
    return data_train_resampleds  # Return the dictionary of resampled datasets


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
    lists = []  # List to store dataset information

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
def run(result_file_path, data_names, classifiers, methods, noise_rate, ntree, is_binary):
    """
    Run the experiments on the specified datasets using the given classifiers and resampling methods.
    
    Parameters:
    - result_file_path: Path to save the results.
    - data_names: List of dataset names.
    - classifiers: List of classifiers to use.
    - methods: List of resampling methods to apply.
    - noise_rate: The noise rate applied to the datasets.
    - ntree: Number of trees for RSDS_zhou.
    - is_binary: Whether the dataset is binary (True) or multi-class (False).
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

    exception_dataset = []  # List to store datasets that caused exceptions

    # Loop through each dataset
    for d, data_name in enumerate(data_names):
        try:
            # Determine the imbalance rate for the dataset
            imb = imbs[data_name] if data_name in imbs.keys() else (
                0 if data_name in uci_extended else 10)
            
            # Load the dataset based on whether it is binary or multi-class
            if is_binary:  # Binary datasets
                data_train, data_test, counters = load_2classes_data(
                    data_name, noise_rate=noise_rate, imbalance_rate=imb)
            else:  # Multi-class datasets
                data_train, data_test, counters = load_k_classes_data_lm(
                    data_name, noise_rate=noise_rate)
            
            # Print dataset information
            print('\n\nnoise rate: {:.2f}  data set {:2d}/{:2d}: {:15s}  samples: {:6d}   features: {:3d}   '
                  'train 0: {:6d}   train 1: {:6d}   test 0: {:6d}   test 1: {:6d}\n'
                  .format(noise_rate, d + 1, len(data_names), data_name,
                          counters[0], counters[1], counters[2], counters[3], counters[4], counters[5]))

            # Apply resampling methods to the training data
            data_train_resampleds = resamplings(data_train, methods=methods, ntree=ntree)

            # Loop through each classifier
            for classifier in classifiers:
                print('    classifier: {:10s}'.format(classifier))
                clf = get_classifier(classifier)  # Get the classifier object
                
                # Loop through each resampling method
                for method in methods:
                    data_train_those = data_train_resampleds['data_train_' + method]  # Extract the resampled dataset
                    
                    # Perform 5-fold cross-validation
                    metri = []  # List to store metrics for each fold
                    kf = ShuffleSplit(n_splits=5)  # Initialize shuffle split
                    
                    for train_index, validate_index in kf.split(data_train_those[0]):
                        train, validate = data_train_those[0][train_index], data_train_those[0][validate_index]
                        try:
                            metrics = get_metrics(clf, train, validate)  # Get evaluation metrics
                        except Exception as e:
                            continue  # Skip this fold if an error occurs
                        del metrics['tfpn']
                        del metrics['right_rate_min_score']
                        metri.append(metrics)

                    # Aggregate metrics across folds
                    for i in range(1, len(metri)):
                        for k, v in metri[i].items():
                            if k in metri[0]:
                                metri[0][k] += v

                    # Average the metrics over the 5 folds
                    for k, v in metri[0].items():
                        metri[0][k] = v / len(metri)

                    metrics = metri[0]  # Final averaged metrics
                    
                    # Record the results to the CSV file
                    excel_line = [
                        data_name, classifier, method, noise_rate,
                        counters[0], counters[1], counters[2], counters[3], counters[4], counters[5],
                        Counter(train[:, 0])[0], Counter(train[:, 0])[1],
                        metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['auc'],
                        metrics['f1'], metrics['g_mean'], now_time_str()
                    ]
                    write_excel(result_file_path, [], table_head=table_head)  # Write table headers
                    print('        {}  {:20s}'.format(now_time_str(), method), metrics)  # Print metrics
                    write_excel(result_file_path, [excel_line], table_head=table_head)  # Write results

        except Exception as e:
            print('------------------------------', data_name, ':\t', e)  # Log the error
            exception_dataset.append(data_name)  # Add the dataset to the exception list
            continue  # Continue to the next dataset
    
    print('&*******&', exception_dataset)  # Print the list of datasets that caused exceptions


# Function to run experiments and compute standard deviation
def run_std(result_file_path_mean, result_file_path_std, data_names, classifiers, methods, noise_rate, ntree, is_binary):
    """
    Run the experiments and compute the mean and standard deviation of the metrics.
    
    Parameters:
    - result_file_path_mean: Path to save the mean results.
    - result_file_path_std: Path to save the standard deviation results.
    - data_names: List of dataset names.
    - classifiers: List of classifiers to use.
    - methods: List of resampling methods to apply.
    - noise_rate: The noise rate applied to the datasets.
    - ntree: Number of trees for RSDS_zhou.
    - is_binary: Whether the dataset is binary (True) or multi-class (False).
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

    exception_dataset = []  # List to store datasets that caused exceptions

    # Loop through each dataset
    for d, data_name in enumerate(data_names):
        try:
            # Determine the imbalance rate for the dataset
            imb = imbs[data_name] if data_name in imbs.keys() else (
                0 if data_name in uci_extended else 10)
            
            # Load the dataset based on whether it is binary or multi-class
            if is_binary:  # Binary datasets
                data_train, data_test, counters = load_2classes_data(
                    data_name, noise_rate=noise_rate, imbalance_rate=imb)
            else:  # Multi-class datasets
                data_train, data_test, counters = load_k_classes_data_lm(
                    data_name, noise_rate=noise_rate)
            
            # Print dataset information
            print('\n\nnoise rate: {:.2f}  data set {:2d}/{:2d}: {:15s}  samples: {:6d}   features: {:3d}   '
                  'train 0: {:6d}   train 1: {:6d}   test 0: {:6d}   test 1: {:6d}\n'
                  .format(noise_rate, d + 1, len(data_names), data_name,
                          counters[0], counters[1], counters[2], counters[3], counters[4], counters[5]))

            # Apply resampling methods to the training data
            data_train_resampleds = resamplings(data_train, methods=methods, ntree=ntree)

            # Loop through each classifier
            for classifier in classifiers:
                print('    classifier: {:10s}'.format(classifier))
                clf = get_classifier(classifier)  # Get the classifier object
                
                # Loop through each resampling method
                for method in methods:
                    data_train_those = data_train_resampleds['data_train_' + method]  # Extract the resampled dataset
                    
                    # Perform 10-fold cross-validation
                    metri = []  # List to store metrics for each fold
                    kf = StratifiedShuffleSplit(n_splits=10)  # Initialize stratified shuffle split
                    
                    for train_index, validate_index in kf.split(data_train_those[0][:, 1:], data_train_those[0][:, 0]):
                        train, validate = data_train_those[0][train_index], data_train_those[0][validate_index]
                        try:
                            metrics = get_metrics(clf, train, validate)  # Get evaluation metrics
                        except Exception as e:
                            raise e  # Raise the exception to stop execution
                        metri.append(metrics)

                    # Compute mean and standard deviation of the metrics
                    metric_std = {'accuracy': [], 'precision': [], 'recall': [], 'auc': [], 'f1': [], 'g_mean': []}
                    for i in range(0, len(metri)):
                        for k, v in metri[i].items():
                            if k in metric_std:
                                metric_std[k].append(v)

                    for k, v in metric_std.items():
                        metri[0][k] = np.mean(v)  # Compute mean
                        metri[1][k] = np.std(v)  # Compute standard deviation

                    # Write mean results to CSV
                    metrics = metri[0]  # Mean metrics
                    excel_line = [
                        data_name, classifier, method, noise_rate,
                        counters[0], counters[1], counters[2], counters[3], counters[4], counters[5],
                        Counter(train[:, 0])[0], Counter(train[:, 0])[1],
                        metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['auc'],
                        metrics['f1'], metrics['g_mean'], now_time_str()
                    ]
                    write_excel(result_file_path_mean, [], table_head=table_head)  # Write table headers
                    print('        {}  {:20s}'.format(now_time_str(), method), metrics)  # Print metrics
                    write_excel(result_file_path_mean, [excel_line], table_head=table_head)  # Write results

                    # Write standard deviation results to CSV
                    metrics = metri[1]  # Standard deviation metrics
                    excel_line = [
                        data_name, classifier, method, noise_rate,
                        counters[0], counters[1], counters[2], counters[3], counters[4], counters[5],
                        Counter(train[:, 0])[0], Counter(train[:, 0])[1],
                        metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['auc'],
                        metrics['f1'], metrics['g_mean'], now_time_str()
                    ]
                    write_excel(result_file_path_std, [], table_head=table_head)  # Write table headers
                    print('        {}  {:20s}'.format(now_time_str(), method), metrics)  # Print metrics
                    write_excel(result_file_path_std, [excel_line], table_head=table_head)  # Write results

        except Exception as e:
            print('------------------------------', data_name, ':\t', e)  # Log the error
            exception_dataset.append(data_name)  # Add the dataset to the exception list
            raise e  # Raise the exception to stop execution
    
    print('&*******&', exception_dataset)  # Print the list of datasets that caused exceptions


# Main function to run experiments
def main(is_binary):
    """
    Main function to run experiments on binary or multi-class datasets.
    
    Parameters:
    - is_binary: Whether the dataset is binary (True) or multi-class (False).
    """
    # Define datasets based on whether they are binary or multi-class
    if is_binary:
        data_names = [
            'ecoli', 'heart', 'pima', 'wine', 'creditApproval', 'wisconsin', 'breastcancer',
            'messidor_features', 'vehicle2', 'vehicle', 'yeast1', 'Faults', 'segment',
            'seismic-bumps', 'wilt', 'mushrooms', 'page-blocks0', 'letter',
            'Data_for_UCI_named', 'avila', 'magic'
        ]
    else:
        data_names = [
            'nuclear', 'contraceptive', 'satimage', 'sensorReadings', 'frogs'
        ]

    # Define classifiers to use
    classifiers = ['KNN', 'LR', 'SVM', 'DTree', 'XGBoost', 'LightGBM', 'AdaBoost', 'GBDT', 'BPNN']
    
    # Define resampling methods to apply
    methods = (
        'smote-SW', 'kmeans-smote-SW', 'ADASYN-SW', 'SMOTE_ENN-SW', 'SMOTE_TomekLinks-SW', 'SMOTE_IPF-SW'
    )

    # Define the result file paths for mean and standard deviation
    result_file_path_mean = r'KBsomte/SMOTE/result table/{}_{} result lq.xls'.format(
        'mean', now_time_str(colon=False))
    result_file_path_std = r'KBsomte/SMOTE/result table/{}_{} result lq.xls'.format(
        'std', now_time_str(colon=False))

    # Run experiments for different noise rates
    run_std(result_file_path_mean, result_file_path_std, data_names, classifiers,
            methods, noise_rate=0.0, ntree=10, is_binary=is_binary)
    run_std(result_file_path_mean, result_file_path_std, data_names, classifiers,
            methods, noise_rate=0.1, ntree=10, is_binary=is_binary)
    run_std(result_file_path_mean, result_file_path_std, data_names, classifiers,
            methods, noise_rate=0.2, ntree=10, is_binary=is_binary)


# Entry point of the script
if __name__ == '__main__':
    main(is_binary=1)  # Run binary classification experiments
    # main(is_binary=0)  # Run multi-class classification experiments