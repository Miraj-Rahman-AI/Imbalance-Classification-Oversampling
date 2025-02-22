# Import necessary libraries
from experiment_v5_NanRd import (  # Import custom functions for data loading, model evaluation, etc.
    load_2classes_data, load_k_classes_data_lm, get_classifier, get_metrics,
    write_excel, now_time_str, sampling_methods, mean_std
)
import json  # For handling JSON data
from NaN import NaN_RD  # Import NaN_RD for handling missing data
from sklearn.model_selection import StratifiedShuffleSplit  # For stratified cross-validation
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
from collections import Counter  # For counting occurrences of elements
import warnings  # To suppress warnings
warnings.filterwarnings('ignore')  # Ignore warnings to keep the output clean


# Function to perform resampling on the training data
def resamplings(data_train, methods, sampling_rate: float):
    """
    Perform resampling on the training data using specified methods.
    
    Parameters:
    - data_train: The training dataset (numpy array).
    - methods: List of resampling methods to apply.
    - sampling_rate: The rate at which to sample the minority class.
    
    Returns:
    - data_train_resampleds: Dictionary containing resampled datasets for each method.
    """
    data_train_resampleds = {}  # Dictionary to store resampled datasets
    
    # Apply NaN_RD to handle missing data
    NData, weight, nans = NaN_RD(data_train)  # NData: original data, weight: relative density, nans: natural neighbors
    
    # Separate majority and minority classes
    data_more = data_train[data_train[:, 0] == 0]  # Majority class
    data_less = data_train[data_train[:, 0] == 1]  # Minority class
    total_num = len(data_more) - len(data_less)  # Total number of samples to generate

    # Loop through each resampling method
    for method in methods:
        data_train_resampleds['data_train_' + method] = []  # Initialize list for the current method
        
        # Apply the resampling method to the training data
        data_train_resampled = sampling_methods(
            data_train=data_train,  # Original data
            method=method,
            weight=weight,
            nans=nans
        )
        print('Oversampled:\t\t', method)  # Print the method being applied
        
        # Extract the minority class samples from the resampled data
        data_train_less = data_train_resampled[data_train_resampled[:, 0] == 1]
        diff = data_train_less[len(data_less):, :]  # Newly generated minority samples
        
        # Randomly select samples based on the sampling rate
        _temp = [i for i in range(len(data_less))]
        index = np.random.choice(_temp, int(total_num * sampling_rate))
        data_less_new = diff[index]  # Selected minority samples
        
        # Combine the original data with the newly generated minority samples
        data_train_resampled = np.vstack((data_train, data_less_new))
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
    columns = ['data_name', 'samples', 'Features', 'Maj', 'Min', 'IR']
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
        lists.append([data_name, data.shape[0], data.shape[1] - 1, dicts[0.0], dicts[1.0], round(float(dicts[0.0] / dicts[1.0]), 2)])

    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(lists, columns=columns, index=None)
    df.to_csv(r'KBsomte\SMOTE\data_infomation/data_info_' + str(noise_rate) + '.csv', index=False)


# Function to gather information about multi-class datasets
def data_info_lm(data_names: list, noise_rate: int):
    """
    Gather information about multi-class datasets, such as the number of samples, features, and class distribution.
    
    Parameters:
    - data_names: List of dataset names.
    - noise_rate: The noise rate applied to the datasets.
    
    Returns:
    - A CSV file containing the dataset information.
    """
    # Define columns for the output CSV
    columns = ['data_name', 'samples', 'Features', 'Maj', 'Min', 'IR']
    lists = []  # List to store dataset information

    # Loop through each dataset and gather information
    for d, data_name in enumerate(data_names):
        # Load the dataset
        data_train, data_test, counters = load_k_classes_data_lm(
            data_name, noise_rate=noise_rate)
        
        # Combine training and test data
        data = np.vstack((data_train, data_test))
        
        # Count the number of samples in each class
        dicts = Counter(data[:, 0])
        
        # Append the dataset information to the list
        lists.append([data_name, data.shape[0], data.shape[1] - 1, dicts[0.0], dicts[1.0], round(float(dicts[0.0] / dicts[1.0]), 2)])

    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(lists, columns=columns, index=None)
    df.to_csv(r'KBsomte\SMOTE\data_infomation/data_info_lm2' + str(noise_rate) + '.csv', index=False)


# Function to run the experiments
def run(result_file_path, data_names, classifiers, methods, noise_rate, is_binary: bool, is_raise: bool = False, sampling_rate: float = 1):
    """
    Run the experiments on the specified datasets using the given classifiers and resampling methods.
    
    Parameters:
    - result_file_path: Path to save the results.
    - data_names: List of dataset names.
    - classifiers: List of classifiers to use.
    - methods: List of resampling methods to apply.
    - noise_rate: The noise rate applied to the datasets.
    - is_binary: Whether the dataset is binary (True) or multi-class (False).
    - is_raise: Whether to raise exceptions or log them.
    - sampling_rate: The rate at which to sample the minority class.
    """
    # Define the table headers for the results
    table_head = ['data', 'Classifier', 'sampling method', 'noise rate', 'samples', 'features', 'train 0',
                  'train 1', 'test 0', 'test 1', 'sampled 0', 'sampled 1', 'accuracy', 'precision', 'recall', 'AUC',
                  'f1', 'g_mean', 'ap', 'right_rate_min', 'tp', 'fp', 'fn', 'tn', 'now_time']

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
        if is_binary:  # Binary datasets
            data_train, data_test, counters = load_2classes_data(
                data_name, noise_rate=noise_rate, imbalance_rate=imb)
        else:  # Multi-class datasets
            data_train, data_test, counters = load_k_classes_data_lm(
                data_name, noise_rate=noise_rate)
        
        # Combine training and test data
        data_train = np.vstack([data_train, data_test])

        # Skip datasets with fewer than 1000 samples
        if counters[0] < 1000:
            continue

        # Print dataset information
        print('\n\nnoise rate: {:.2f} sampling rate: {:.2f} data set {:2d}/{:2d}: {:15s}  samples: {:6d}   features: {:3d}   '
              'train 0: {:6d}   train 1: {:6d}   test 0: {:6d}   test 1: {:6d}\n'
              .format(noise_rate, sampling_rate, d + 1, len(data_names), data_name,
                      counters[0], counters[1], counters[2], counters[3], counters[4], counters[5]))

        # Apply resampling methods to the training data
        try:
            data_train_resampleds = resamplings(data_train, methods=methods, sampling_rate=sampling_rate)
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

                # Perform 5-fold cross-validation
                metri = Counter({})  # Dictionary to store metrics
                skf = StratifiedShuffleSplit(n_splits=5)  # Initialize stratified shuffle split
                
                try:
                    for train_index, validate_index in skf.split(data_train_those[0][:, 1:], data_train_those[0][:, 0]):
                        train, validate = data_train_those[0][train_index], data_train_those[0][validate_index]
                        metrics = get_metrics(clf, train, validate)  # Get evaluation metrics
                        metri += Counter(metrics)  # Aggregate metrics across folds
                    
                    # Average the metrics over the 5 folds
                    metrics = {k: round(v / 5, 5) for k, v in metri.items()}

                    # Record the results to the CSV file
                    excel_line = [
                        data_name, classifier, method, noise_rate,
                        counters[0], counters[1], counters[2], counters[3], counters[4], counters[5],
                        Counter(train[:, 0])[0], Counter(train[:, 0])[1],
                        metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['auc'],
                        metrics['f1'], metrics['g_mean'], metrics['ap'], now_time_str()
                    ]
                    write_excel(result_file_path, [], table_head=table_head)  # Write table headers
                    print('{}  {:20s}'.format(now_time_str(), method), metrics)  # Print metrics
                    write_excel(result_file_path, [excel_line], table_head=table_head)  # Write results
                except Exception as e:
                    if is_raise:
                        raise e
                    else:
                        debug.append([noise_rate, data_name, method, classifier, str(e)])

    # Save debug information to a CSV file
    df = pd.DataFrame(debug)
    df.to_csv(r'KBsomte/SMOTE/Log/' + '.csv', index=False)


# Main function to run experiments
def main(is_binary: bool, is_raise: bool = False):
    """
    Main function to run experiments on binary or multi-class datasets.
    
    Parameters:
    - is_binary: Whether the dataset is binary (True) or multi-class (False).
    - is_raise: Whether to raise exceptions or log them.
    """
    # Define datasets based on whether they are binary or multi-class
    if is_binary:
        data_names = [
            'ecoli-0_vs_1', 'new-thyroid1', 'ecoli', 'creditApproval', 'wisconsin', 'breastcancer',
            'messidor_features', 'vehicle2', 'vehicle', 'yeast1', 'Faults', 'segment', 'seismic-bumps',
            'wilt', 'mushrooms', 'page-blocks0', 'Data_for_UCI_named', 'avila', 'magic'
        ]
    else:
        data_names = [
            'nuclear', 'contraceptive', 'satimage'
        ]

    # Define classifiers to use
    classifiers = [
        'KNN', 'DTree', 'XGBoost', 'LightGBM', 'AdaBoost', 'GBDT'
    ]

    # Define resampling methods to apply
    methods = (
        'smote', 'smote-Nan', 'borderline1-smote', 'SMOTE_TomekLinks', 'SMOTE_IPF', 'ADASYN', 'DBSMOTE', 'AdaptiveSMOTE', 'GDO'
    )

    # Define the result file path
    result_file_path = r'KBsomte/SMOTE/result table/binary/{} NanRd.xls'.format(
        now_time_str(colon=False))

    # Run experiments for different noise rates and sampling rates
    for n_r in np.arange(0, 0.3, 0.1):
        for s_r in np.arange(0.1, 0.5, 0.1):
            run(result_file_path, data_names, classifiers, methods, noise_rate=n_r, is_binary=is_binary, is_raise=is_raise, sampling_rate=s_r)


# Entry point of the script
if __name__ == '__main__':
    # main(is_binary=0)  # Run multi-class classification experiments
    main(is_binary=1, is_raise=0)  # Run binary classification experiments