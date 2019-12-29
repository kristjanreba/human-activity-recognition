import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from utils import load_data


def CV(x, y):
    accs = []
    kf = KFold(n_splits=5, random_state=1, shuffle=True)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = SVC(random_state=1, gamma='scale', kernel='rbf', C=1000)
        clf.fit(x_train, y_train)
        train_acc = clf.score(x_train, y_train)
        test_acc = clf.score(x_test, y_test)
        print('SVC train accuracy: ', clf.score(x_train, y_train))
        print('SVC test accuracy: ', clf.score(x_test, y_test))
        accs.append(test_acc)

    std = np.std(accs)
    mean = np.mean(accs)
    print('SVM acc = {} +/- {}'.format(mean, std))


if __name__ == '__main__':
    x, y = load_data()
    #CV(x, y)