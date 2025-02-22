# Import necessary libraries
import math  # For mathematical operations
import numpy as np  # For numerical computations
import pandas as pd  # For data manipulation and analysis
from collections import Counter  # For counting occurrences of elements
from sklearn.model_selection import StratifiedShuffleSplit  # For stratified cross-validation
import warnings  # To suppress warnings
warnings.filterwarnings('ignore')  # Ignore warnings to keep the output clean

# Custom module imports
from RSDS import RSDS_zhou  # Import RSDS_zhou for data denoising and weighting
from experiment_Framework import (  # Import custom functions for data loading, model evaluation, etc.
    load_2classes_data, load_k_classes_data_lm, get_classifier, get_metrics,
    write_excel, now_time_str, sampling_methods, mean_std
)
from NaN import NaN_RD  # Import NaN_RD for handling missing data


# Function to perform resampling on the training data
def resamplings(data_train, methods):
    """
    Perform resampling on the training data using specified methods.
    
    Parameters:
    - data_train: The training dataset (numpy array).
    - methods: List of resampling methods to apply.
    
    Returns:
    - data_train_resampleds: Dictionary containing resampled datasets for each method.
    """
    data_train_resampleds = {}  # Dictionary to store resampled datasets
    
    # Calculate the number of trees for RSDS_zhou based on the dataset size
    ntree = int(math.log(data_train.shape[0] * data_train.shape[1] - 1, 2) / 2)
    
    # Apply RSDS_zhou to denoise and weight the dataset
    RData, weight_SW = RSDS_zhou(data_train, ntree)  # RData: denoised dataset, weight_SW: weight matrix
    
    # Apply NaN_RD to handle missing data
    NData, weight_WNND, nans = NaN_RD(data_train)  # NData: original data, weight_WNND: relative density, nans: natural neighbors

    # Loop through each resampling method
    for method in methods:
        data_train_resampleds['data_train_' + method] = []  # Initialize list for the current method
        
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
        print('Oversampled:\t\t', method)  # Print the method being applied
        
        # Store the resampled data in the dictionary
        data_train_resampleds['data_train_' + method].append(data_train_resampled)
    
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
def run(result_file_path, data_names, classifiers, methods, noise_rate, is_binary: bool):
    """
    Run the experiments on the specified datasets using the given classifiers and resampling methods.
    
    Parameters:
    - result_file_path: Path to save the results.
    - data_names: List of dataset names.
    - classifiers: List of classifiers to use.
    - methods: List of resampling methods to apply.
    - noise_rate: The noise rate applied to the datasets.
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

        # Print dataset information
        print('\n\nnoise rate: {:.2f}  data set {:2d}/{:2d}: {:15s}  samples: {:6d}   features: {:3d}   '
              'train 0: {:6d}   train 1: {:6d}   test 0: {:6d}   test 1: {:6d}\n'
              .format(noise_rate, d + 1, len(data_names), data_name,
                      counters[0], counters[1], counters[2], counters[3], counters[4], counters[5]))

        # Apply resampling methods to the training data
        try:
            data_train_resampleds = resamplings(data_train, methods=methods)
        except Exception as e:
            debug.append([noise_rate, data_name, str(e)])  # Log the error
            raise e  # Raise the exception to stop execution

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
                    debug.append([noise_rate, data_name, method, classifier, str(e)])  # Log the error
                    raise e  # Raise the exception to stop execution

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
                        metrics['f1'], metrics['g_mean'], now_time_str()
                    ]

                    write_excel(result_file_path, [], table_head=table_head)  # Write table headers
                    print('{}  {:20s}'.format(now_time_str(), method), metrics)  # Print metrics
                    write_excel(result_file_path, [excel_line], table_head=table_head)  # Write results
                except Exception as e:
                    raise e  # Raise the exception to stop execution
                    debug.append([noise_rate, data_name, method, classifier, str(e)])  # Log the error

    # Save debug information to a CSV file
    df = pd.DataFrame(debug)
    df.to_csv(r'KBsomte\debug' + str(noise_rate) + '.csv', index=False)


# Main function to run experiments
def main(is_binary: bool):
    """
    Main function to run experiments on binary or multi-class datasets.
    
    Parameters:
    - is_binary: Whether the dataset is binary (True) or multi-class (False).
    """
    # Define datasets based on whether they are binary or multi-class
    if is_binary:
        data_names = [
            'ecoli-0_vs_1', 'new-thyroid1', 'ecoli', 'creditApproval', 'wisconsin', 'breastcancer',
            'messidor_features', 'vehicle2', 'vehicle', 'yeast1', 'Faults', 'segment', 'seismic-bumps',
            'wilt', 'mushrooms', 'Data_for_UCI_named', 'avila', 'magic'
        ]
    else:
        data_names = [
            'nuclear', 'contraceptive', 'satimage', 'sensorReadings', 'frogs'
        ]

    # Define classifiers to use
    classifiers = [
        'KNN', 'DTree', 'XGBoost', 'LightGBM', 'AdaBoost', 'GBDT'
    ]

    # Define resampling methods to apply
    methods = (
        'SMOTE-Nan', 'SMOTE-W', 'SMOTE-G', 'SMOTE-SW', 'SMOTE',
        'SMOTE_ENN-GB', 'SMOTE_ENN-W', 'SMOTE_ENN-G', 'SMOTE_ENN', 'SMOTE_ENN-SW',
        'SMOTE_TomekLinks-GB', 'SMOTE_TomekLinks-W', 'SMOTE_TomekLinks-G', 'SMOTE_TomekLinks', 'SMOTE_TomekLinks-SW',
        'SMOTE_RSB-GB', 'SMOTE_RSB-W', 'SMOTE_RSB-G', 'SMOTE_RSB', 'SMOTE_RSB-SW',
        'SMOTE_IPF-GB', 'SMOTE_IPF-W', 'SMOTE_IPF-G', 'SMOTE_IPF', 'SMOTE_RSB-SW'
    )

    # Define the result file path
    result_file_path = r'KBsomte/SMOTE/result table/{} result Gsmote.xls'.format(
        now_time_str(colon=False))

    # Run the experiments with the specified noise rate
    run(result_file_path, data_names, classifiers, methods, noise_rate=0.1, is_binary=is_binary)


# Entry point of the script
if __name__ == '__main__':
    main(is_binary=1)  # Run binary classification experiments
    main(is_binary=0)  # Run multi-class classification experiments