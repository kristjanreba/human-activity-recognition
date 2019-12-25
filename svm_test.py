import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from utils import load_data, data_info


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    '''
    Cs = np.logspace(-6, 3, 10)
    parameters = [{'kernel': ['rbf'], 'C': Cs},
                {'kernel': ['linear'], 'C': Cs}]
    '''

    Cs = [1000]
    parameters = [{'kernel': ['rbf'], 'C': Cs}]

    svc = SVC(random_state=1, gamma='scale')
    clf = GridSearchCV(estimator=svc, param_grid=parameters, cv=5, n_jobs=-1)
    clf.fit(x_train.values, y_train.values.flatten())
    #print(clf.best_params_)
    #print(clf.best_score_)
    print('SVC train accuracy: ', clf.score(x_train, y_train))
    print('SVC test accuracy: ', clf.score(x_test, y_test))

    # Majority class classifier
    acc = y_test.value_counts().values[0] / y_test.value_counts().values.sum()
    print('Majority class classifier accuracy:', acc)