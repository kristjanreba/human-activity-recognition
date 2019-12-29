import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from utils import load_data


def CV(x, y):
    accs = []
    kf = KFold(n_splits=5, random_state=1, shuffle=True)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = MLPClassifier(solver='adam',
                            alpha=1e-4,
                            hidden_layer_sizes=(32, 32, 16),
                            max_iter=2000,
                            random_state=1)

        clf.fit(x_train, y_train)
        train_acc = clf.score(x_train, y_train)
        test_acc = clf.score(x_test, y_test)
        print('MLP train accuracy: ', train_acc)
        print('MLP test accuracy: ', test_acc)
        accs.append(test_acc)

    std = np.std(accs)
    mean = np.mean(accs)
    print('MLP acc = {} +/- {}'.format(mean, std))



if __name__ == '__main__':
    x, y = load_data()
    CV(x, y)

    '''
    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    sn.heatmap(cm, annot=True)
    plt.show()
    '''
