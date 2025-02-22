# Import necessary libraries
from experiment_Framework import (
    load_2classes_data,  # Function to load binary classification datasets
    load_k_classes_data_lm,  # Function to load multi-class datasets
    get_classifier,  # Function to get a classifier based on the name
    get_metrics,  # Function to compute evaluation metrics
    write_excel,  # Function to write results to an Excel file
    now_time_str,  # Function to get the current time as a string
    sampling_methods,  # Function to apply various sampling methods
    mean_std  # Function to compute mean and standard deviation
)
from collections import Counter  # For counting occurrences of elements
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import StratifiedShuffleSplit  # For stratified sampling
import warnings  # To handle warnings
import math  # For mathematical operations
import logging  # For logging errors and information
from RSDS import RSDS_zhou  # Custom resampling method

# Suppress warnings to avoid clutter in the output
warnings.filterwarnings('ignore')

# Define a function to apply resampling methods to the training data
def resamplings(data_train, methods):
    """
    Apply various resampling methods to the training data.

    Parameters:
    - data_train: The training dataset.
    - methods: A list of resampling methods to apply.

    Returns:
    - data_train_resampleds: A dictionary containing resampled datasets for each method.
    """
    data_train_resampleds = {}
    ntree = int(math.log(data_train.shape[0] * data_train.shape[1] - 1, 2) / 2)  # Calculate the number of trees for RSDS

    # Apply the RSDS_zhou method to denoise and weight the data
    RData, weight = RSDS_zhou(data_train, ntree)

    # Iterate over each resampling method
    for method in methods:
        data_train_resampleds['data_train_' + method] = []
        # Apply the resampling method to the training data
        data_train_resampled = sampling_methods(data_train=data_train,
                                                method=method,
                                                weight=weight,
                                                ntree=ntree,
                                                RData=RData)
        print('Oversampled:\t\t', method)
        data_train_resampleds['data_train_' + method].append(data_train_resampled)

    return data_train_resampleds

# Define a function to gather information about the datasets
def data_info(data_names: list, noise_rate: int):
    """
    Gather information about the datasets, such as the number of samples, features, and class distribution.

    Parameters:
    - data_names: A list of dataset names.
    - noise_rate: The noise rate to apply to the datasets.

    Returns:
    - A CSV file containing the dataset information.
    """
    uci_extended = ['breast_tissue2', 'ecoli', 'glass', 'haberman', 'heart', 'iris', 'libra', 'liver_disorders',
                    'pima', 'segment', 'vehicle', 'wine', 'simulated1', 'simulated2', 'simulated3', 'simulated4',
                    'simulated5', 'simulated6', 'simulated7', 'simulated8', 'simulated9', 'simulated10']
    imbs = {'splice': 5, 'breastcancer': 5, 'creditApproval': 5, 'votes': 5, 'Online_Retail': 50,
            'codrna': 20, 'nomao': 20, 'mocap': 40, 'poker': 50, 'skin': 100, 'covtype': 60, 'comedy': 100}
    columns = ['data_name', 'samples', 'Features', 'Maj', 'Min']
    lists = []

    # Iterate over each dataset
    for d, data_name in enumerate(data_names):
        imb = imbs[data_name] if data_name in imbs.keys() else (0 if data_name in uci_extended else 10)
        # Load the dataset
        data_train, data_test, counters = load_2classes_data(data_name, noise_rate=noise_rate, imbalance_rate=imb)
        data = np.vstack((data_train, data_test))
        dicts = Counter(data[:, 0])
        lists.append([data_name, data.shape[0], data.shape[1] - 1, dicts[0.0], dicts[1.0]])

    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(lists, columns=columns, index=None)
    df.to_csv(r'KBsomte\SMOTE\data_infomation/data_info_' + str(noise_rate) + '.csv', index=False)

# Define the main function to run the experiments
def run(result_file_path, data_names, classifiers, methods, noise_rate, is_binary):
    """
    Run the experiments with different datasets, classifiers, and resampling methods.

    Parameters:
    - result_file_path: The path to save the results.
    - data_names: A list of dataset names.
    - classifiers: A list of classifiers to use.
    - methods: A list of resampling methods to apply.
    - noise_rate: The noise rate to apply to the datasets.
    - is_binary: A flag to indicate whether the datasets are binary or multi-class.

    Returns:
    - Results saved in an Excel file.
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
        imb = imbs[data_name] if data_name in imbs.keys() else (0 if data_name in uci_extended else 10)
        if is_binary:
            data_train, data_test, counters = load_2classes_data(data_name, noise_rate=noise_rate, imbalance_rate=imb)
        else:
            data_train, data_test, counters = load_k_classes_data_lm(data_name, noise_rate=noise_rate)

        data_train = np.vstack([data_train, data_test])

        if counters[0] < 1000:
            continue  # Skip datasets with fewer than 1000 samples

        print('\n\nnoise rate: {:.2f}  data set {:2d}/{:2d}: {:15s}  samples: {:6d}   features: {:3d}   '
              'train 0: {:6d}   train 1: {:6d}   test 0: {:6d}   test 1: {:6d}\n'
              .format(noise_rate, d + 1, len(data_names), data_name,
              counters[0], counters[1], counters[2], counters[3], counters[4], counters[5]))

        try:
            data_train_resampleds = resamplings(data_train, methods=methods)
        except Exception as e:
            debug.append([noise_rate, data_name, str(e)])
            raise e

        for classifier in classifiers:
            print('    classifier: {:10s}'.format(classifier))
            clf = get_classifier(classifier)
            for method in methods:
                try:
                    data_train_those = data_train_resampleds['data_train_' + method]
                except Exception as e:
                    debug.append([noise_rate, data_name, method, classifier, str(e)])
                    raise e

                metri = Counter({})
                skf = StratifiedShuffleSplit(n_splits=5)

                try:
                    for train_index, validate_index in skf.split(data_train_those[0][:, 1:], data_train_those[0][:, 0]):
                        train, validate = data_train_those[0][train_index], data_train_those[0][validate_index]
                        metrics = get_metrics(clf, train, validate)
                        metri += Counter(metrics)
                    metrics = {k: round(v / 5, 5) for k, v in metri.items()}

                    excel_line = [
                        data_name, classifier, method, noise_rate,
                        counters[0], counters[1], counters[2], counters[3], counters[4], counters[5],
                        Counter(train[:, 0])[0], Counter(train[:, 0])[1],
                        metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['auc'],
                        metrics['f1'], metrics['g_mean'], now_time_str()
                    ]

                    write_excel(result_file_path, [], table_head=table_head)
                    print('{}  {:20s}'.format(now_time_str(), method), metrics)
                    write_excel(result_file_path, [excel_line], table_head=table_head)
                except Exception as e:
                    debug.append([noise_rate, data_name, method, classifier, str(e)])
                    raise e

    # Save debug information to a CSV file
    df = pd.DataFrame(debug)
    df.to_csv(r'KBsomte\debug' + '.csv', index=False)

# Define the main function to set up and run the experiments
def main(is_binary: bool):
    """
    Main function to set up and run the experiments.

    Parameters:
    - is_binary: A flag to indicate whether to use binary or multi-class datasets.
    """
    if is_binary:
        data_names = [
            'ecoli-0_vs_1', 'new-thyroid1', 'ecoli',
            'creditApproval', 'wisconsin', 'breastcancer',
            'messidor_features', 'vehicle2', 'vehicle',
            'yeast1', 'Faults',
            'segment', 'seismic-bumps',
            'wilt', 'mushrooms',
            'page-blocks0', 'Data_for_UCI_named',
            'avila', 'magic',
        ]
    else:
        data_names = [
            'nuclear', 'contraceptive', 'satimage', 'sensorReadings', 'frogs'
        ]

    classifiers = [
        'KNN', 'DTree',
        'XGBoost', 'LightGBM',
        'AdaBoost', 'GBDT'
    ]

    methods = (
        'SMOTE-GB', 'SMOTE-W', 'SMOTE-G', 'SMOTE', 'SMOTE-SW',
        'SMOTE_ENN-GB', 'SMOTE_ENN-W', 'SMOTE_ENN-G', 'SMOTE_ENN', 'SMOTE_ENN-SW',
        'SMOTE_TomekLinks-GB', 'SMOTE_TomekLinks-W', 'SMOTE_TomekLinks-G', 'SMOTE_TomekLinks', 'SMOTE_TomekLinks-SW',
        'SMOTE_RSB-GB', 'SMOTE_RSB-W', 'SMOTE_RSB-G', 'SMOTE_RSB', 'SMOTE_RSB-SW',
        'SMOTE_IPF-GB', 'SMOTE_IPF-W', 'SMOTE_IPF-G', 'SMOTE_IPF', 'SMOTE_RSB-SW',
    )

    result_file_path = r'KBsomte/SMOTE/result table/{} result Gsmote.xls'.format(now_time_str(colon=False))

    run(result_file_path, data_names, classifiers, methods, noise_rate=0.05, is_binary=is_binary)
    run(result_file_path, data_names, classifiers, methods, noise_rate=0.15, is_binary=is_binary)
    run(result_file_path, data_names, classifiers, methods, noise_rate=0.25, is_binary=is_binary)

# Entry point of the script
if __name__ == '__main__':
    main(is_binary=1)  # Run the experiments with binary datasets