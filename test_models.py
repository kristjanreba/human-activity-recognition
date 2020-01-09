import pickle
import warnings
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from time import time
from datetime import datetime
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.optimize import least_squares
from scipy.stats import mode
from imblearn.over_sampling import RandomOverSampler, SMOTE

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from utils import load_data


def cross_valid(X, y, model, oversampling):
    accs = []
    k_folds = StratifiedKFold(n_splits=5, shuffle=True)
    for train_indexs, test_indexs in k_folds.split(X,y):
        X_train, X_test = X[train_indexs], X[test_indexs]
        y_train, y_test = y[train_indexs], y[test_indexs]

        if oversampling:
            sm = SMOTE(random_state=0)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        
        y_train_predict, y_test_predict = model(X_train, y_train, X_test)
        acc = accuracy_score(y_test, y_test_predict)
        accs.append(acc)
    mean = np.round(np.mean(accs), 6)
    std = np.round(np.std(accs), 6)
    print("Mean error: " + str(mean) + " +/- " + str(std) + "%")


def LGBM(X_train, y_train, X_test):
    train_dataset = lgb.Dataset(X_train, label=y_train)

    params = {}
    params['verbose'] = -1
    params['objective'] = 'multiclass'
    params['num_class'] = 9
    #params['importance_type'] = 'gain'
    #params['num_iterations'] = 200
    #params['learning_rate'] = 0.05
    params['boosting_type'] = 'gbdt'
    #params['metric'] = 'mse'
    #params['dart'] = True,
    #params['sub_feature'] = 0.5
    params['num_leaves'] = 64
    #params['min_data_in_leaf'] = 30
    params['max_depth'] = 10
    params['max_bin'] = 64
    #params['bagging_fraction'] = 0.5
    #params['bagging_freq'] = 32

    model = lgb.train(params, train_dataset, 200)
    y_train_predict = np.argmax(model.predict(X_train), axis=1)
    y_test_predict = np.argmax(model.predict(X_test), axis=1)
    return y_train_predict, y_test_predict


def SVM(X_train, y_train, X_test):
    model = svm.SVC(C = 1e4,
                    kernel = 'rbf',
                    cache_size=1000)

    model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)
    return y_train_predict, y_test_predict


def DNN(X_train, y_train, X_test):
    model = MLPClassifier(hidden_layer_sizes = (60,40,30,),
                            activation ='relu',
                            solver ='adam',
                            alpha = 1e-5,
                            batch_size = 'auto',
                            learning_rate ='constant',
                            max_iter = int(1e5),
                            n_iter_no_change = 50,
                            verbose=True)

    model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)
    return y_train_predict, y_test_predict


def KNN(X_train, y_train, X_test):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)
    return y_train_predict, y_test_predict


def RFC(X_train, y_train, X_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)
    return y_train_predict, y_test_predict


def ETC(X_train, y_train, X_test):
    model = ExtraTreesClassifier()
    model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)
    return y_train_predict, y_test_predict


def Ensamble(X_train, y_train, X_test):
    train_pred = np.zeros((y_train.shape[0], 3))
    test_pred = np.zeros((X_test.shape[0], 3))
    y_train_predict, y_test_predict = LGBM(X_train, y_train, X_test)
    train_pred[:,0] = y_train_predict
    test_pred[:,0] = y_test_predict
    y_train_predict, y_test_predict = KNN(X_train, y_train, X_test)
    train_pred[:,1] = y_train_predict
    test_pred[:,1] = y_test_predict
    y_train_predict, y_test_predict = ETC(X_train, y_train, X_test)
    train_pred[:,2] = y_train_predict
    test_pred[:,2] = y_test_predict
    y_train_predict = mode(train_pred, axis=1)[0]
    y_test_predict = mode(test_pred, axis=1)[0]
    return y_train_predict, y_test_predict


if __name__ == '__main__':
    window_size = 1
    CV = False
    oversampling = False
    test_size = 0.33

    #prediction_models = [LGBM, DNN]
    #model_names = ['LGBM', 'DNN']
    #prediction_models = [LGBM, KNN, RFC, ETC]
    #model_names = ['LGBM', 'KNN', 'RFC', 'ETC']

    prediction_models = [LGBM, KNN]
    model_names = ['LGBM', 'KNN']

    X, y = load_data(window_size=window_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    print('Oversampling: {}'.format(oversampling))
    print('Window size: {}'.format(window_size))

    if oversampling:
        sm = SMOTE(random_state=0)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    for model, model_name in zip(prediction_models, model_names):
        print('Model: {}'.format(model_name))
        
        if CV: cross_valid(X, y, model, oversampling)
        else:
            start_time = time()
            y_train_predict, y_test_predict = model(X_train, y_train, X_test)
            end_time = time()

            print("Time taken for training and testing: {0:.4f} s".format(end_time - start_time))
            print('Train accuracy: {}'.format(accuracy_score(y_train, y_train_predict)))
            print('Test accuracy: {}'.format(accuracy_score(y_test, y_test_predict)))

            '''
            b = y_test != y_test_predict
            y_test = y_test[b]
            y_test_predict = y_test_predict[b]
            cm = confusion_matrix(y_test, y_test_predict)
            sn.heatmap(cm, annot=True)
            plt.show()
            '''
