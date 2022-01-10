import sys

import pandas as pd
import numpy as np
import os
from pathlib import Path
import time

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from TSC_utils import *
from MLP import *
from TSF_MLP import *
from TSF_CNN import *
from GAF_CNN import *
from FCNN import *
from ResNet import *
from RNN import *
from LSTMFCN import *
from CNN_RNN import *
from CNN_MLP_Autoencoder import *
from RNN_Autoencoder import *
from classifiers_dataset_info import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score



# List of data sets
dataset_names = [
                'mag_cohorts_2021_v3',
                 'shape_cohorts_2021_v3']
                # 'local_shape_cohorts_2021']



path = os.path.abspath(os.getcwd()) + '/'
save_path = path + 'results/'
os.chdir(path)
#======================================================================================================
def run_main_code(sel, n_runs = 1, slice_ratio= [0.7, 0.15, 0.15], verbose= False):
    # Data scaling method
    scale =   scale = ['normalize', 'normalize', 'min_max', 'normalize', 'normalize', 'normalize', 'normalize',
             'normalize', 'normalize', 'normalize', 'normalize', 'normalize']


    print(classifiers_name[sel] + ' is running ...')
    seed = 1368
    #======================================================================================================
    # Main Program - Model Training and Testing
    df_metrics = []
    df_outputs = []
    init_acc_eff = np.empty([0, 4])    # record accuracy over random initialization
    auc_report = np.empty([0, 4])
    for dataset_name in dataset_names:
        print(f'Dataset: {dataset_name}')
        nb_classes = dataset_dict[dataset_name]['nb_classes']
        root_path = dataset_dict[dataset_name]['path']
        directory_names = directory_contents(root_path)
        for dirs in directory_names:
            run_path = root_path + dirs
            fnames = directory_contents(run_path, 1)
            for fname in fnames:
                print(fname)

                # strs = fname.split('_')
                # if (strs[6] != '0') or (strs[5] != '0.25'):
                #     continue

                X, y, ID_Class = read_dataset(run_path, fname, dataset_dict, dataset_name, scale = scale[sel])
                x_train, y_train, x_val, y_val, x_test, y_test, indexes = train_test_dataset(X, y, seed, slice_ratio)
                save_name = dataset_name + '_' + dirs + fname[17:-4]
                print(run_path + " / " + fname)

                MC_out = np.empty([0, 6])
                # Hold model outputs on train and test data sets during Monte Carlo runs
                out_res_train = np.empty([n_runs, y_train.shape[0], y_train.shape[1]])
                out_res_val = np.empty([n_runs, y_val.shape[0], y_val.shape[1]])
                out_res_test = np.empty([n_runs, y_test.shape[0], y_test.shape[1]])
                for run in range(n_runs):
                    print('MC Run #' + str(run))
                    if sel == 0:  # select MLP
                        y_train_pred, y_val_pred, y_test_pred, duration, training_itrs = MLP_classifier(save_path, save_name,
                                                                                                        x_train, y_train, x_val, y_val, x_test, y_test,
                                                                                                        nb_classes, run, verbose, min_exp_val_loss=0.02)
                    elif sel == 1:  # select TSF_MLP
                        y_train_pred, y_val_pred, y_test_pred, duration, training_itrs = TSF_MLP_classifier(save_path, save_name,
                                                                                                            x_train, y_train, x_val, y_val, x_test, y_test,
                                                                                                            nb_classes, run, verbose, min_exp_val_loss=0.02)
                    elif sel == 2:  # Select GAF_CNN
                        if run == 0:
                            x_train = gramian_angular_field(x_train)
                            x_test = gramian_angular_field(x_test)
                            x_val = gramian_angular_field(x_val)
                            # X_GAF = np.concatenate((x_train, x_test), axis=0)
                            # Y_GAF = np.concatenate((y_train, y_test), axis=0)
                            # GAF_viz(X_GAF, Y_GAF, save_path, save_name, dataset_name, dataset_dict)  # GAF Visualization
                        y_train_pred, y_val_pred, y_test_pred, duration, training_itrs = GAF_CNN_classifier(save_path, save_name,
                                                                                                            x_train, y_train, x_val, y_val, x_test, y_test,
                                                                                                            nb_classes, run, verbose, min_exp_val_loss=0.02)
                    elif sel == 3:  # Select FCNN
                        y_train_pred, y_val_pred, y_test_pred, duration, training_itrs = FCNN_classifier(save_path, save_name,
                                                                                                         x_train, y_train, x_val, y_val, x_test, y_test,
                                                                                                         nb_classes, run, verbose, min_exp_val_loss=0.02)
                    elif sel == 4:  # Select ResNet
                        y_train_pred, y_val_pred, y_test_pred, duration, training_itrs = ResNet_classifier(save_path, save_name,
                                                                                                           x_train, y_train, x_val, y_val, x_test, y_test,
                                                                                                           nb_classes, run, verbose, min_exp_val_loss=0.02)
                    elif sel == 5:  # Select RNN
                        y_train_pred, y_val_pred, y_test_pred, duration, training_itrs = RNN_classifier(save_path, save_name,
                                                                                                        x_train, y_train, x_val, y_val, x_test, y_test,
                                                                                                        nb_classes, run, verbose, min_exp_val_loss=0.02)
                    elif sel == 6:  # Select CNN-RNN
                        y_train_pred, y_val_pred, y_test_pred, duration, training_itrs = CNN_RNN_classifier(save_path, save_name,
                                                                                                            x_train, y_train, x_val, y_val, x_test, y_test,
                                                                                                            nb_classes, run, verbose, min_exp_val_loss=0.02)
                    elif sel == 7:  # Select CNN_MLP_Autoencoder
                        y_train_pred, y_val_pred, y_test_pred, duration, training_itrs = CNN_MLP_Autoencoder(save_path, save_name,
                                                                                                             x_train, y_train, x_val, y_val, x_test, y_test,
                                                                                                             nb_classes, run, verbose, min_exp_val_loss=0.02)
                    elif sel == 8:  # Select RNN_Autoencoder
                        y_train_pred, y_val_pred, y_test_pred, duration, training_itrs = RNN_Autoencoder(save_path, save_name,
                                                                                                         x_train, y_train, x_val, y_val, x_test, y_test,
                                                                                                         nb_classes, run, verbose, min_exp_val_loss=0.02)
                    elif sel == 10:  # select TSF_CNN
                        y_train_pred, y_val_pred, y_test_pred, duration, training_itrs = TSF_CNN_classifier(save_path, save_name, x_train, y_train, x_val,
                                                                                                            y_val,
                                                                                                            x_test, y_test, nb_classes, run, verbose,
                                                                                                            min_exp_val_loss=0.02)
                    elif sel == 12:  # Variational Autoencoder
                        y_train_pred, y_val_pred, y_test_pred, duration, training_itrs = Var_Auto_classifier(save_path, save_name,
                                                                                                             X1_train, X2_train, y_train, X1_val, X2_val, y_val,
                                                                                                             X1_test, X2_test, y_test, nb_classes, run, verbose,
                                                                                                             min_exp_val_loss=0.02)
                    elif sel == 11:  # Select LSTM-FCN
                        y_train_pred, y_val_pred, y_test_pred, duration, training_itrs = LSTMFCN_classifier(save_path, save_name,
                                                                                                        x_train, y_train, x_val, y_val, x_test, y_test,
                                                                                                        nb_classes, run, verbose, min_exp_val_loss=0.02)

                    else:
                        sys.exit('Error!')
                    # Decision threshold is calculated based on Youden’s J statistic.
                    # J = Sensitivity + (1 – FalsePositiveRate) – 1
                    # Which we can restate as:
                    # J = TruePositiveRate – FalsePositiveRate
                    fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1), y_test_pred[:, 1], pos_label=1)
                    test_AUC = roc_auc_score(np.argmax(y_test, axis=1), y_test_pred[:, 1])
                    # if test_AUC < 0.4:
                    #     raise Exception(f"Test AUC is wierd! : {test_AUC}")

                    # get the best threshold
                    ix = np.argmax(tpr - fpr)
                    best_thresh = thresholds[ix] if thresholds[ix] < 1 else thresholds[ix+1]
                    if 1 < best_thresh < 0:
                        raise Exception(f"Decision threshold issue! : {best_thresh}")
                    acc_best = accuracy_score(np.argmax(y_test, axis=1), (y_test_pred[:, 1] >= best_thresh).astype("int"))

                    fpr, tpr, thresholds = roc_curve(np.argmax(y_train, axis=1), y_train_pred[:, 1], pos_label=1)
                    train_AUC = roc_auc_score(np.argmax(y_train, axis=1), y_train_pred[:, 1])
                    print(f'Train AUC:{round(train_AUC,2)}, Test AUC: {round(test_AUC,2)}, Optimal threshold: {best_thresh}')
                    MC_out = np.vstack((MC_out, np.array([train_AUC, test_AUC, best_thresh, acc_best, duration, training_itrs])))
                    out_res_train[run] = y_train_pred
                    out_res_val[run] = y_val_pred
                    out_res_test[run] = y_test_pred
                median_index = np.argsort(MC_out[:, 0])[len(MC_out[:, 0])//2]
                init_acc_eff = np.vstack((init_acc_eff, np.array([np.mean(MC_out[:, 0]), np.median(MC_out[:, 0]),
                                                                  np.min(MC_out[:, 0]), np.max(MC_out[:, 0])])))
                MC_out = pd.DataFrame(MC_out, columns=['train_AUC', 'test_AUC', 'best_thresh', 'acc_best', 'duration', 'training_itrs'])
                for run in range(n_runs):
                    y_val_pred = out_res_val[run]
                    y_test_pred = out_res_test[run]
                    y_train_pred = out_res_train[run]
                    cohort_path_name = '/' + dataset_name + '/' + dirs + '/' + fname
                    METRICS = calculate_metrics(y_train, y_train_pred, y_test, y_test_pred, save_name, MC_out.iloc[run, :])
                    df_metrics.append(METRICS)
                    if run == median_index:
                        src = save_path + 'models/' + save_name + '_' + classifiers_name[sel] + '_history_run_' + str(run) + '.csv'
                        dst = save_path + 'models/' + save_name + '_' + classifiers_name[sel] + '_history.csv'
                        os.rename(src, dst)
                        src = save_path + 'models/' + save_name + '_' + classifiers_name[sel] + '_best_model_run_' + str(run) + '.hdf5'
                        dst = save_path + 'models/' + save_name + '_' + classifiers_name[sel] + '_best_model.hdf5'
                        os.rename(src, dst)
                        OUTPUTS = prepare_outputs(np.concatenate((y_train, y_val)), np.concatenate((y_train_pred, y_val_pred)), y_test, y_test_pred, duration,
                                              training_itrs,
                                              indexes, ID_Class, seed, cohort_path_name, classifiers_name[sel],
                                              dataset_name, dataset_dict)  # method_dict,
                        df_outputs.append(OUTPUTS)
                    else:
                        src = save_path + 'models/' + save_name + '_' + classifiers_name[sel] + '_best_model_run_' + str(run) + '.hdf5'
                        os.remove(src)
                        src = save_path + 'models/' + save_name + '_' + classifiers_name[sel] + '_history_run_' + str(run) + '.csv'
                        os.remove(src)


    df_metrics = pd.concat(df_metrics,axis=0, sort=False)
    df_outputs = pd.concat(df_outputs, axis=0, sort=False)
    pd.DataFrame(init_acc_eff, columns=['MEAN', 'MED', 'MIN', 'MAX']).to_csv(save_path + classifiers_name[sel] + '_initialization_effect.csv', index = False)
    df_metrics.to_csv(save_path + classifiers_name[sel] + '_metrics.csv', index = False)
    df_outputs.to_csv(save_path + classifiers_name[sel] + '_outputs.csv', index = False)
#======================================================================================================
if __name__ == "__main__":
    # List of classifiers
    slice_ratio = [0.7, 0.15, 0.15]
    assert np.sum(slice_ratio) == 1
    classifiers_name = ['MLP', 'TSF_MLP', 'GAF_CNN', 'FCNN', 'ResNet', 'RNN', 'CNN_RNN', 'CNN_MLP_Autoencoder', 'RNN_Autoencoder', 'Logreg_GAF_CNN', 'TSF_CNN', 'LSTMFCN']
    verbose=False

    print("Enter a number:")
    for i in range(len(classifiers_name)):
        print(f"{i}: {classifiers_name[i]}")
    num = int(input("Which Classifier (0 to 11)?"))
    core = input("Which GPU (0 to 7)?")

    # Number of Monte-Carlo runs
    n_runs = int(input("Number of run?"))

    # n_runs=10
    # Select GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(core)

    start = time.time()
    run_main_code(num, n_runs= n_runs, slice_ratio= slice_ratio, verbose=verbose)
    print(f"Classifier: {classifiers_name[num]} Elapsed time:  {time.time() - start}")
    # for i in range(9,len(classifiers_name)):
    #     print(f"{i}: {classifiers_name[i]}")
    #     start = time.time()
    #     run_main_code(i, n_runs= n_runs, slice_ratio= slice_ratio)
    #     print(f"Classifier: {classifiers_name[i]} Elapsed time:  {time.time()-start}")
#======================================================================================================
