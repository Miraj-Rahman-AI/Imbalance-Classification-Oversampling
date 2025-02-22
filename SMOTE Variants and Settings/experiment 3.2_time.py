import datetime
from experiment_v5_NanRd import (load_2classes_data, load_k_classes_data_lm, get_classifier, get_metrics,
                              write_excel, now_time_str, sampling_methods, mean_std)
import json
from NaN import NaN_RD
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def resamplings(res_file_path, data_train, methods, data_name):
    data_train_resampleds = []

    for method in methods:
        # data_train_resampleds = []
        t1 = datetime.datetime.now()
        if 'Nan' in method:
            NData, weight, nans = NaN_RD(data_train)    
            # print(t1.strftime("%Y-%m-%d %H:%M:%S.%f"))
            data_train_resampled = sampling_methods(data_train=data_train,  
                                                    method=method,
                                                    weight=weight,
                                                    nans=nans)
        else:
             data_train_resampled = sampling_methods(data_train=data_train,  
                                                    method=method,
                                                    weight=None,
                                                    nans=None)
        t2 = datetime.datetime.now()
        # print(t2.strftime("%Y-%m-%d %H:%M:%S.%f"))
        cha = str((t2-t1).seconds)+'.'+str((t2-t1).microseconds)    # str
        print(method, "Change time:", cha)
        data_train_resampleds.append([data_name, method, cha])
    # print('data_train_resampleds', type(
    #     data_train_resampleds), data_train_resampleds)
    write_excel(res_file_path, data_train_resampleds)


def run_time(result_file_path, data_names, methods, noise_rate, is_binary: bool):
    table_head = ['data', 'Classifier', 'sampling method',
                  'noise rate', 'samples', 'features', 'time']

    uci_extended = ['breast_tissue2', 'ecoli', 'glass', 'haberman', 'heart', 'iris', 'libra', 'liver_disorders',
                    'pima', 'segment', 'vehicle', 'wine', 'simulated1', 'simulated2', 'simulated3', 'simulated4',
                            'simulated5', 'simulated6', 'simulated7', 'simulated8', 'simulated9', 'simulated10']
    imbs = {'splice': 5, 'breastcancer': 5, 'creditApproval': 5, 'votes': 5, 'Online_Retail': 50,
            'codrna': 20, 'nomao': 20, 'mocap': 40, 'poker': 50, 'skin': 100, 'covtype': 60, 'comedy': 100}
    time_res = []

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
            resamplings(result_file_path, data_train,
                        methods=methods, data_name=data_name)
        except Exception as e:
            debug.append([noise_rate, data_name, str(e)])
            # raise e
            continue
            # write_excel(result_file_path, [], table_head=table_head)
            # print('{}  {:20s}'.format(now_time_str(), method), metrics)

    # save debug json
    df = pd.DataFrame(debug)
    df.to_csv(r'debug'+str(noise_rate)+'.csv', index=False)


def main(is_binary: bool):
    if is_binary:
        data_names = [
            # 20000
            # 'ecoli-0_vs_1',
            # 'new-thyroid1',
            # 'ecoli',
            # 'creditApproval', 'wisconsin', 'diabetes', 'breastcancer',
            # 'messidor_features', 'vehicle2', 'vehicle',
            # 'yeast1', 'Faults',
            # 'segment','svmguide1', 'seismic-bumps',
            # 'wilt',
            # 'mushrooms',
            # # 'page-blocks0',
            # 'Data_for_UCI_named','letter', 'avila','magic',

            'ecoli-0_vs_1', 'new-thyroid1', 'ecoli', 'creditApproval', 'wisconsin',  'breastcancer',
            'messidor_features', 'vehicle2', 'vehicle', 'yeast1', 'Faults', 'segment', 'seismic-bumps',
            'wilt', 'mushrooms', 'Data_for_UCI_named', 
            'avila', 'magic', 'nomao',


            # 数据量20000以上
            # 'susy', 'default of credit card clients', 
            # 'mocap','poker',
        ]
    else:
        data_names = [
            'nuclear',
            'contraceptive',
            'satimage',
            'sensorReadings', 'frogs',

        ]

    classifiers = [
        # 'KNN', 'DTree',
        # # 'LR',
        # 'XGBoost', 'LightGBM',
        # # 'SVM',
        # 'AdaBoost',
        # 'GBDT',
        'LR', 'SVM', 'BPNN',
    ]

    # imbalanced_learn without '-'
    # smote_variants  with  '-'
    methods = (
        'smote','smote-Nan',                         # pre
        'borderline1-smote', 'boderline1-smote-Nan',
        'SVM_balance', 'SVM_balance-Nan',
        'MSYN_', 'MSYN_-Nan',

        'SMOTE_ENN','SMOTE_ENN-Nan',                  # post
        'SMOTE_TomekLinks', 'SMOTE_TomekLinks-Nan',
        # 'SMOTE_RSB','SMOTE_RSB-Nan',
        'SMOTE_IPF', 'SMOTE_IPF-Nan',
        'SMOTE_FRST_2T','SMOTE_FRST_2T-Nan',


        # 'smote', 'borderline1-smote',     # one vs others
        # 'SMOTE_TomekLinks', 'SMOTE_IPF', 'SMOTE_FRST_2T',

        'ADASYN', 'DBSMOTE',
        # 'Gaussian_SMOTE',
        'AdaptiveSMOTE', 'GDO',
        # # 'MSYN_',
        # # 'LDAS',

        # 'RSMOTE',
        # 'kmeans-smote',
        # 'SMOTE-GB', 'MWMOTE', 'DBSMOTE','kmeans-smote','RSMOTE',
        # 'SMOTE_TomekLinks', 'SMOTE_IPF','SMOTE_FRST_2T',
        # 'AdaptiveSMOTE','ADASYN', 'GDO',
    )

    # for n_r in range(0, 7):
    #     # print(n_r)
    #     run(result_file_path, data_names,classifiers, methods, noise_rate=n_r*0.05, is_binary=is_binary)

    # data_info(data_names, 0, 1)
    
    result_file_path = r'KBsomte/SMOTE/result table/{}_{} result time.xls'.format(
        'mean', now_time_str(colon=False)) 
    run_time(result_file_path, data_names, methods,
             noise_rate=0.0, is_binary=is_binary)
    
    result_file_path = r'KBsomte/SMOTE/result table/{}_{} result time.xls'.format(
        'mean', now_time_str(colon=False)) 
    run_time(result_file_path, data_names, methods,
             noise_rate=0.0, is_binary=is_binary)
    
    result_file_path = r'KBsomte/SMOTE/result table/{}_{} result time.xls'.format(
        'mean', now_time_str(colon=False)) 
    run_time(result_file_path, data_names, methods,
             noise_rate=0.0, is_binary=is_binary)
    
    result_file_path = r'KBsomte/SMOTE/result table/{}_{} result time.xls'.format(
        'mean', now_time_str(colon=False)) 
    run_time(result_file_path, data_names, methods,
             noise_rate=0.0, is_binary=is_binary)
    
    result_file_path = r'KBsomte/SMOTE/result table/{}_{} result time.xls'.format(
        'mean', now_time_str(colon=False)) 
    run_time(result_file_path, data_names, methods,
             noise_rate=0.0, is_binary=is_binary)
    
    result_file_path = r'KBsomte/SMOTE/result table/{}_{} result time.xls'.format(
        'mean', now_time_str(colon=False)) 
    run_time(result_file_path, data_names, methods,
             noise_rate=0.0, is_binary=is_binary)
    
    result_file_path = r'KBsomte/SMOTE/result table/{}_{} result time.xls'.format(
        'mean', now_time_str(colon=False)) 
    run_time(result_file_path, data_names, methods,
             noise_rate=0.0, is_binary=is_binary)
    
    result_file_path = r'KBsomte/SMOTE/result table/{}_{} result time.xls'.format(
        'mean', now_time_str(colon=False)) 
    run_time(result_file_path, data_names, methods,
             noise_rate=0.0, is_binary=is_binary)
    
    result_file_path = r'KBsomte/SMOTE/result table/{}_{} result time.xls'.format(
        'mean', now_time_str(colon=False)) 
    run_time(result_file_path, data_names, methods,
             noise_rate=0.0, is_binary=is_binary)
    
    result_file_path = r'KBsomte/SMOTE/result table/{}_{} result time.xls'.format(
        'mean', now_time_str(colon=False)) 
    run_time(result_file_path, data_names, methods,
             noise_rate=0.0, is_binary=is_binary)

    # run(result_file_path, data_names, classifiers,
    #     methods, noise_rate=0.0, is_binary=is_binary)
    # run(result_file_path, data_names, classifiers,
    #     methods, noise_rate=0.0, is_binary=is_binary)


if __name__ == '__main__':
    # main(is_binary=0)    
    main(is_binary=1)  
