from experiment_v5_NanRd import (load_2classes_data, load_k_classes_data_lm, get_classifier, get_metrics,
                              write_excel, now_time_str, sampling_methods, mean_std,sampling_methods_General)
import json
from NaN import NaN_RD
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
import math
from collections import Counter
from RSDS import RSDS_zhou
import warnings
warnings.filterwarnings('ignore')


def resamplings(data_train, methods,k_neighbors):
    data_train_resampleds = {}
    ntree = int(math.log(data_train.shape[0]*data_train.shape[1]-1,2)/2)
    print("ntree**************************************:\t",ntree)
    RData,weight = RSDS_zhou(data_train, ntree)      
    
    for method in methods:
        data_train_resampleds['data_train_' + method] = []
        data_train_resampled = sampling_methods_General(data_train=data_train,  
                                                method=method,k_neighbors=k_neighbors)
        print('Oversampled:\t\t', method)
        data_train_resampleds['data_train_' +
                              method].append(data_train_resampled)
    return data_train_resampleds


def data_info(data_names: list, noise_rate: int, is_binary: bool):
    ''' 获取数据集信息 '''
    uci_extended = ['breast_tissue2', 'ecoli', 'glass', 'haberman', 'heart', 'iris', 'libra', 'liver_disorders',
                    'pima', 'segment', 'vehicle', 'wine', 'simulated1', 'simulated2', 'simulated3', 'simulated4',
                            'simulated5', 'simulated6', 'simulated7', 'simulated8', 'simulated9', 'simulated10']
    imbs = {'splice': 5, 'breastcancer': 5, 'creditApproval': 5, 'votes': 5, 'Online_Retail': 50,
            'codrna': 20, 'nomao': 20, 'mocap': 40, 'poker': 50, 'skin': 100, 'covtype': 60, 'comedy': 100}
    columns = ['data_name', 'samples', 'Features', 'Maj', 'Min']
    lists = []

    for d, data_name in enumerate(data_names):
        imb = imbs[data_name] if data_name in imbs.keys() else (
            0 if data_name in uci_extended else 10)
        if is_binary:
            data_train, data_test, counters = load_2classes_data(
                data_name, noise_rate=noise_rate, imbalance_rate=imb)  
        else:   
            data_train, data_test, counters = load_k_classes_data_lm(
                data_name, noise_rate=noise_rate)  

        data = np.vstack((data_train, data_test))
        dicts = Counter(data[:, 0])
        lists.append([data_name, data.shape[0],
                      data.shape[1]-1, dicts[0.0], dicts[1.0]])

    df = pd.DataFrame(lists, columns=columns, index=None)

    df.to_csv(r'KBsomte\SMOTE\data_infomation/data_info_' + str(is_binary) +
              str(noise_rate) + '.csv', index=False)


def run(result_file_path, data_names, classifiers, methods, noise_rate, is_binary: bool):
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
        if is_binary:   
            data_train, data_test, counters = load_2classes_data(
                data_name, noise_rate=noise_rate, imbalance_rate=imb)  
        else:   
            data_train, data_test, counters = load_k_classes_data_lm(
                data_name, noise_rate=noise_rate)  

        data_train = np.vstack([data_train, data_test])

        # if counters[0] > 20000:continue  # TODO:
        print('\n\nnoise rate: {:.2f}  data set {:2d}/{:2d}: {:15s}  samples: {:6d}   features: {:3d}   '
              'train 0: {:6d}   train 1: {:6d}   test 0: {:6d}   test 1: {:6d}\n'
              .format(noise_rate, d + 1, len(data_names), data_name,
                      counters[0], counters[1], counters[2], counters[3], counters[4], counters[5]))

        try:
            data_train_resampleds = resamplings(data_train, methods=methods)
        except Exception as e:
            debug.append([noise_rate, data_name, str(e)])
            raise e
            continue

        for classifier in classifiers:
            print('    classifier: {:10s}'.format(classifier))
            clf = get_classifier(classifier)
            for method in methods:
                
                try:
                    data_train_those = data_train_resampleds['data_train_' + method]
                except Exception as e:
                    debug.append(
                        [noise_rate, data_name, method, classifier, str(e)])
                    raise e
                    continue

                # 5 fold cross validation   
                metri = Counter({})
                skf = StratifiedShuffleSplit(n_splits=5)

                try:
                    for train_index, validate_index in skf.split(data_train_those[0][:, 1:], data_train_those[0][:, 0]):
                        train, validate = data_train_those[0][train_index], data_train_those[0][validate_index]
                        metrics = get_metrics(clf, train, validate)
                        metri += Counter(metrics)
                    metrics = {k: round(v/5, 5)
                               for k, v, in metri.items()}   

                    # record the results to csv
                    excel_line = [
                        data_name, classifier, method, noise_rate,
                        counters[0], counters[1], counters[2], counters[3], counters[4], counters[5],
                        Counter(train[:, 0])[0], Counter(train[:, 0])[1],
                        metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['auc'],
                        metrics['f1'], metrics['g_mean'], now_time_str()
                    ]

                    write_excel(result_file_path, [], table_head=table_head)
                    print('{}  {:20s}'.format(now_time_str(), method), metrics)
                    write_excel(result_file_path, [
                                excel_line], table_head=table_head)
                except Exception as e:
                    debug.append(
                        [noise_rate, data_name, method, classifier, str(e)])
                    raise e
                    continue

    # save debug json
    df = pd.DataFrame(debug)
    df.to_csv(r'debug'+str(noise_rate)+'.csv', index=False)

def run_std(result_file_path_mean, result_file_path_std, data_names, classifiers, methods, noise_rate, is_binary,k_neighbors):
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
        if is_binary:   
            data_train, data_test, counters = load_2classes_data(
                data_name, noise_rate=noise_rate, imbalance_rate=imb)  
        else:   
            data_train, data_test, counters = load_k_classes_data_lm(
                data_name, noise_rate=noise_rate)  

        data_train = np.vstack([data_train, data_test])

        # if counters[0] < 1000:
        #     continue  # TODO:
        print('\n\nnoise rate: {:.2f}  data set {:2d}/{:2d}: {:15s}  samples: {:6d}   features: {:3d}   '
              'train 0: {:6d}   train 1: {:6d}   test 0: {:6d}   test 1: {:6d}\n'
              .format(noise_rate, d + 1, len(data_names), data_name,
                      counters[0], counters[1], counters[2], counters[3], counters[4], counters[5]))

        try:
            data_train_resampleds = resamplings(data_train, methods=methods,k_neighbors=k_neighbors)
        except Exception as e:
            debug.append([noise_rate, data_name, str(e)])
            # continue
            raise e

        for classifier in classifiers:
            print('    classifier: {:10s}'.format(classifier))
            clf = get_classifier(classifier)
            for method in methods:
                
                try:
                    data_train_those = data_train_resampleds['data_train_' + method]
                except Exception as e:
                    debug.append(
                        [noise_rate, data_name, method, classifier, str(e)])
                    # continue
                    raise e

                # 10 fold cross validation
                metri = []
                kf = StratifiedShuffleSplit(n_splits=10)

                for train_index, validate_index in kf.split(data_train_those[0][:, 1:], data_train_those[0][:, 0]):
                    train, validate = data_train_those[0][train_index], data_train_those[0][validate_index]
                    try:
                        metrics = get_metrics(
                            clf, train, validate)  
                    except Exception as e:
                        # continue
                        raise e
                    # del metrics['tfpn']
                    # del metrics['right_rate_min_score']
                    metri.append(metrics)

                metric_std = {'accuracy': [], 'precision': [],
                              'recall': [], 'auc': [], 'f1': [], 'g_mean': []}
                for i in range(0, len(metri)):  
                    for k, v in metri[i].items():
                        if k in metric_std:
                            metric_std[k].append(v)

                for k, v in metric_std.items():  
                    metri[0][k] = np.mean(v)
                    metri[1][k] = np.std(v)

                metrics = metri[0]  
                excel_line = [data_name, classifier, method, noise_rate,
                              counters[0], counters[1], counters[2], counters[3], counters[4], counters[5],
                              Counter(train[:, 0])[0], Counter(
                                  train[:, 0])[1],
                              metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['auc'],
                              metrics['f1'], metrics['g_mean'],
                              now_time_str()]
                write_excel(result_file_path_mean, [],
                            table_head=table_head)  # todo
                print('        {}  {:20s}'.format(
                    now_time_str(), method), metrics)  # todo
                write_excel(result_file_path_mean, [
                    excel_line], table_head=table_head)  # todo

                metrics = metri[1]  
                excel_line = [data_name, classifier, method, noise_rate,
                              counters[0], counters[1], counters[2], counters[3], counters[4], counters[5],
                              Counter(train[:, 0])[0], Counter(
                                  train[:, 0])[1],
                              metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['auc'],
                              metrics['f1'], metrics['g_mean'],
                              now_time_str()]
                write_excel(result_file_path_std, [],
                            table_head=table_head)  # todo
                print('        {}  {:20s}'.format(
                    now_time_str(), method), metrics)  # todo
                write_excel(result_file_path_std, [
                    excel_line], table_head=table_head)  # todo

    # save debug json
    df = pd.DataFrame(debug)
    df.to_csv(r'KBsomte\debug'+str(noise_rate)+'.csv', index=False)


def main(is_binary: bool):
    if is_binary:
        data_names = [
            # 20000
            # 'ecoli-0_vs_1',
            # 'new-thyroid1',
            'ecoli',
            # 'creditApproval', 'wisconsin', 'breastcancer',
            # 'messidor_features', 'vehicle2', 'vehicle',
            # 'yeast1', 'Faults',
            # 'segment', 'seismic-bumps',
            'wilt',
            # 'mushrooms',
            # # 'page-blocks0',
            # 'Data_for_UCI_named',
            # 'avila',
            #  'magic',

            # 'ecoli-0_vs_1','new-thyroid1', 'ecoli', 
            # 'heart', 'pima',  'wine',
            # 'creditApproval', 'wisconsin', 'breastcancer',
            # 'messidor_features', 'vehicle2', 'vehicle',
            # 'yeast1', 'Faults','segment',
            # 'seismic-bumps', 'wilt', 
            # 'mushrooms',
            # 'page-blocks0', 
            # 'Data_for_UCI_named', 
            # 'avila', 
            # 'magic',
            # 'wisconsin', 

            # 数据量20000以上
            # 选择的大数据集
            # 'default of credit card clients', 'susy', 'nomao', 
            # 'mocap','poker', 'skin',
        ]
    else:
        data_names = [
            'nuclear',
            'contraceptive',
            'satimage',
            'sensorReadings', 
            'frogs',


        ]

    classifiers = [
        'LR', 'KNN','SVM', 'DTree',
        'LightGBM','XGBoost','AdaBoost', 'GBDT',
        'BPNN', 
    ]

    # imbalanced_learn without '-'
    # smote_variants  with  '-'
    methods = (
        'smote',
        'smote-Nan',                         # pre
        # 'borderline1-smote','boderline1-smote-Nan',
        # 'SVM_balance','SVM_balance-Nan',
        # 'MSYN_','MSYN_-Nan',

        # 'SMOTE_ENN','SMOTE_ENN-Nan',                  # post
        # 'SMOTE_TomekLinks','SMOTE_TomekLinks-Nan',
        # # 'SMOTE_RSB','SMOTE_RSB-Nan',
        # 'SMOTE_IPF','SMOTE_IPF-Nan',
        # 'SMOTE_FRST_2T','SMOTE_FRST_2T-Nan',


        # 'smote', 'borderline1-smote', 
        # 'ADASYN', 'MWMOTE','DBSMOTE','RSMOTE','kmeans-smote',

        # 'AdaptiveSMOTE','LDAS','GDO',

        # # 'AdaptiveSMOTE','MSYN_','LDAS','GDO','GBSY',

        # 'SMOTE_TomekLinks', 'SMOTE_IPF','SMOTE_FRST_2T', 'SMOTE_RSB', 'SMOTE_ENN',
        # 'GDO',
    )

    result_file_path_mean = r'KBsomte/SMOTE/result table/{}_{} result Gsmote.xls'.format(
        'mean', now_time_str(colon=False))                                                                    # TODO: result_file_path
    result_file_path_std = r'KBsomte/SMOTE/result table/{}_{} result Gsmote.xls'.format(
        'std', now_time_str(colon=False))                                                                     # TODO: result_file_path
    # for n_r in range(0, 5, 1):
    # run(result_file_path, data_names, classifiers,
    #     methods, noise_rate=n_r*0.05, is_binary=is_binary)

    # run_std(result_file_path_mean, result_file_path_std, data_names, classifiers,
    #     methods, noise_rate=0.0, is_binary=is_binary)
    run_std(result_file_path_mean, result_file_path_std, data_names, classifiers,
        methods, noise_rate=0.0, is_binary=is_binary,k_neighbors=5)
    run_std(result_file_path_mean, result_file_path_std, data_names, classifiers,
        methods, noise_rate=0.1, is_binary=is_binary,k_neighbors=5)
    run_std(result_file_path_mean, result_file_path_std, data_names, classifiers,
        methods, noise_rate=0.2, is_binary=is_binary,k_neighbors=5)


if __name__ == '__main__':
    # main(is_binary=0)    
    main(is_binary=1)  
