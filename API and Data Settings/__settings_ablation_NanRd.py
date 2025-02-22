# Import necessary libraries
from experiment_ablation import (load_2classes_data, load_k_classes_data_lm, get_classifier, get_metrics,
                                 write_excel, now_time_str, sampling_methods, mean_std)
from collections import Counter  # For counting occurrences of elements
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and saving results
from sklearn.model_selection import StratifiedShuffleSplit  # For stratified cross-validation
from NaN import NaN_RD  # Custom module for handling NaN values and resampling
import warnings  # To suppress warnings during execution
warnings.filterwarnings('ignore')  # Ignore all warnings
from ablation_1 import NNk_weight, NaN_SMOTE, Nan_rd_weight, Nan_num_weight  # Custom functions for ablation experiments

# Function to perform resampling on the training data using specified methods
def resamplings(data_train, methods):
    """
    Applies resampling methods to the training data.
    
    Args:
        data_train (numpy.ndarray): The training data.
        methods (list): List of resampling methods to apply.
    
    Returns:
        dict: A dictionary containing resampled datasets for each method.
    """
    data_train_resampleds = {}  # Dictionary to store resampled datasets

    # Apply NaN_RD to the training data to get NData, weight, and nans
    NData, weight, nans = NaN_RD(data_train)
    
    # Apply NNk_weight to get weights and neighbors for 5-NN and 3-NN
    NData, weight_5, neighbors_5 = NNk_weight(data_train, 5)  # 5-NN + weights
    NData, weight_3, neighbors_3 = NNk_weight(data_train, 3)  # 3-NN + weights

    # Iterate over each resampling method
    for method in methods:
        data_train_resampleds['data_train_' + method] = []  # Initialize list for each method
        
        # Apply the sampling method to the training data
        data_train_resampled = sampling_methods(data_train=data_train,  # Use the denoised and weighted data
                                                method=method,
                                                weight=weight,
                                                nans=nans,
                                                weight_5=weight_5, 
                                                neighbors_5=neighbors_5,
                                                weight_3=weight_3, 
                                                neighbors_3=neighbors_3
                                                )
        print('Oversampled:\t\t', method)  # Print the method being applied
        data_train_resampleds['data_train_' + method].append(data_train_resampled)  # Store resampled data
    
    return data_train_resampleds  # Return the dictionary of resampled datasets

# Function to gather information about the datasets
def data_info(data_names: list, noise_rate: int):
    """
    Gathers information about the datasets, such as the number of samples, features, and class distribution.
    
    Args:
        data_names (list): List of dataset names.
        noise_rate (int): Noise rate applied to the datasets.
    """
    # List of datasets from the UCI repository
    uci_extended = ['breast_tissue2', 'ecoli', 'glass', 'haberman', 'heart', 'iris', 'libra', 'liver_disorders',
                    'pima', 'segment', 'vehicle', 'wine', 'simulated1', 'simulated2', 'simulated3', 'simulated4',
                    'simulated5', 'simulated6', 'simulated7', 'simulated8', 'simulated9', 'simulated10']
    
    # Dictionary of datasets with their imbalance rates
    imbs = {'splice': 5, 'breastcancer': 5, 'creditApproval': 5, 'votes': 5, 'Online_Retail': 50,
            'codrna': 20, 'nomao': 20, 'mocap': 40, 'poker': 50, 'skin': 100, 'covtype': 60, 'comedy': 100}
    
    columns = ['data_name', 'samples', 'Features', 'Maj', 'Min']  # Columns for the output CSV
    lists = []  # List to store dataset information

    # Iterate over each dataset name
    for d, data_name in enumerate(data_names):
        # Determine the imbalance rate for the dataset
        imb = imbs[data_name] if data_name in imbs.keys() else (
            0 if data_name in uci_extended else 10)
        
        # Load the dataset with the specified noise and imbalance rates
        data_train, data_test, counters = load_2classes_data(
            data_name, noise_rate=noise_rate, imbalance_rate=imb)  # Load 2-class dataset
        
        data = np.vstack((data_train, data_test))  # Combine training and test data
        dicts = Counter(data[:, 0])  # Count the occurrences of each class
        lists.append([data_name, data.shape[0],
                      data.shape[1]-1, dicts[0.0], dicts[1.0]])  # Append dataset info to the list

    # Create a DataFrame from the dataset information
    df = pd.DataFrame(lists, columns=columns, index=None)

    # Save dataset information to a CSV file
    df.to_csv(r'KBsomte\SMOTE\data_infomation/data_info_' +
              str(noise_rate) + '.csv', index=False)

# Main function to run the experiments
def run(result_file_path, data_names, classifiers, methods, noise_rate, is_binary, is_raise: bool = 0):
    """
    Runs the main experiment loop.
    
    Args:
        result_file_path (str): Path to save the results.
        data_names (list): List of dataset names.
        classifiers (list): List of classifiers to evaluate.
        methods (list): List of resampling methods to apply.
        noise_rate (float): Noise rate applied to the datasets.
        is_binary (bool): Whether the datasets are binary or multi-class.
        is_raise (bool): Whether to raise exceptions or log them.
    """
    # Table headers for the results
    table_head = ['data', 'Classifier', 'sampling method', 'noise rate', 'samples', 'features', 'train 0',
                  'train 1', 'test 0', 'test 1', 'sampled 0', 'sampled 1', 'accuracy', 'precision', 'recall', 'AUC',
                  'f1', 'g_mean', 'right_rate_min', 'tp', 'fp', 'fn', 'tn', 'now_time']

    # List of datasets from the UCI repository
    uci_extended = ['breast_tissue2', 'ecoli', 'glass', 'haberman', 'heart', 'iris', 'libra', 'liver_disorders',
                    'pima', 'segment', 'vehicle', 'wine', 'simulated1', 'simulated2', 'simulated3', 'simulated4',
                    'simulated5', 'simulated6', 'simulated7', 'simulated8', 'simulated9', 'simulated10']
    
    # Dictionary of datasets with their imbalance rates
    imbs = {'splice': 5, 'breastcancer': 5, 'creditApproval': 5, 'votes': 5, 'Online_Retail': 50,
            'codrna': 20, 'nomao': 20, 'mocap': 40, 'poker': 50, 'skin': 100, 'covtype': 60, 'comedy': 100}

    debug = []  # List to log errors during execution
    for d, data_name in enumerate(data_names):
        # Adjust the dataset to achieve the given imbalance rate (uci_extended datasets are not processed)
        imb = imbs[data_name] if data_name in imbs.keys() else (
            0 if data_name in uci_extended else 10)
        
        # Load the dataset based on whether it is binary or multi-class
        if is_binary:   # Binary classification
            data_train, data_test, counters = load_2classes_data(
                data_name, noise_rate=noise_rate, imbalance_rate=imb)  # Load 2-class dataset
        else:   # Multi-class classification
            data_train, data_test, counters = load_k_classes_data_lm(
                data_name, noise_rate=noise_rate)  # Load multi-class dataset

        data_train = np.vstack([data_train, data_test])  # Combine training and test data

        # Print dataset information
        print('\n\nnoise rate: {:.2f}  data set {:2d}/{:2d}: {:15s}  samples: {:6d}   features: {:3d}   '
              'train 0: {:6d}   train 1: {:6d}   test 0: {:6d}   test 1: {:6d}\n'
              .format(noise_rate, d + 1, len(data_names), data_name,
              counters[0], counters[1], counters[2], counters[3], counters[4], counters[5]))

        try:
            # Apply resampling methods to the training data
            data_train_resampleds = resamplings(data_train, methods=methods)
        except Exception as e:
            if is_raise: raise e  # Raise exception if is_raise is True
            else: debug.append([noise_rate, data_name, str(e)])  # Log the error

        # Evaluate each classifier
        for classifier in classifiers:
            print('    classifier: {:10s}'.format(classifier))  # Print the classifier being evaluated
            clf = get_classifier(classifier)  # Get the classifier object
            for method in methods:
                # Extract the resampled dataset from the dictionary
                try:
                    data_train_those = data_train_resampleds['data_train_' + method]
                except Exception as e:
                    if is_raise: raise e  # Raise exception if is_raise is True
                    else: debug.append([noise_rate, data_name, method, classifier, str(e)])  # Log the error

                # Perform 5-fold cross-validation
                metri = Counter({})  # Counter to store metrics
                skf = StratifiedShuffleSplit(n_splits=5)  # Stratified cross-validation

                try:
                    for train_index, validate_index in skf.split(data_train_those[0][:, 1:], data_train_those[0][:, 0]):
                        train, validate = data_train_those[0][train_index], data_train_those[0][validate_index]
                        metrics = get_metrics(clf, train, validate)  # Get evaluation metrics
                        metri += Counter(metrics)  # Accumulate metrics
                    metrics = {k: round(v/5, 5) for k, v in metri.items()}  # Average the metrics over the folds

                    # Record the results to CSV
                    excel_line = [
                        data_name, classifier, method, noise_rate,
                        counters[0], counters[1], counters[2], counters[3], counters[4], counters[5],
                        Counter(train[:, 0])[0], Counter(train[:, 0])[1],
                        metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['auc'],
                        metrics['f1'], metrics['g_mean'], now_time_str()
                    ]

                    write_excel(result_file_path, [], table_head=table_head)  # Write headers to Excel
                    print('{}  {:20s}'.format(now_time_str(), method), metrics)  # Print metrics
                    write_excel(result_file_path, [excel_line], table_head=table_head)  # Write results to Excel
                except Exception as e:
                    if is_raise: raise e  # Raise exception if is_raise is True
                    else: debug.append([noise_rate, data_name, method, classifier, str(e)])  # Log the error

    # Save debug information to a CSV file
    df = pd.DataFrame(debug)
    df.to_csv(r'KBsomte\debug' + '.csv', index=False)

# Main function to set up and run the experiments
def main(is_binary: bool, is_raise=0):
    """
    Sets up the datasets, classifiers, and resampling methods to be used in the experiments.
    
    Args:
        is_binary (bool): Whether the datasets are binary or multi-class.
        is_raise (bool): Whether to raise exceptions or log them.
    """
    if is_binary:
        data_names = [
            # Datasets with less than 20000 samples
            'ecoli-0_vs_1', 'new-thyroid1', 'ecoli', 
            'creditApproval', 'wisconsin', 'breastcancer',
            'messidor_features', 'vehicle2', 'vehicle',
            'yeast1', 'Faults',
            'segment', 'seismic-bumps',
            'wilt', 'mushrooms',
            'page-blocks0', 
            'Data_for_UCI_named',
            'avila', 'magic',

            # Datasets with more than 20000 samples
            # 'nomao',
            # 'poker',

            # 'wisconsin', 
            # 'messidor_features',
            # 'yeast1', 
            # 'segment', 
            # 'wilt', 
            # 'magic',

            'breastcancer', 
        ]
    else:
        data_names = [
            'nuclear', 
            'contraceptive',
             'satimage', 'sensorReadings', 'frogs'
        ]

    classifiers = [
        'KNN', 'DTree',
        # 'LR',
        'XGBoost', 'LightGBM',
        # 'SVM',
        'AdaBoost', 'GBDT'
    ]

    methods = (
        # 'smote', 
        # 'Nan-weight',
        
        # '3nn_weight',   # 3nn + weights
        # '5nn_weight',   # 5nn + weights
        # 'Nan_smote',    # Nan + random sampling
        # 'Nan_rd',       # Nan + density weights
        # 'Nan_num',      # Nan + count weights
        'Nan_weight_rand', # Nan + weight(rd + Nan count) + seed count random + rd position weights #WRND seed random version
    )

    result_file_path = r'KBsomte/SMOTE/result table/{} result NanRd_K.xls'.format(
        now_time_str(colon=False))  # Path to save results

    # Run the experiments for different noise rates
    for n_r in np.arange(0, 0.3, 0.1):
        run(result_file_path, data_names, classifiers, methods, noise_rate=n_r, is_binary=is_binary, is_raise=is_raise)

if __name__ == '__main__':
    # main(is_binary=0)
    main(is_binary=1, is_raise=1)