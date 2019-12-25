import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from utils import load_data, data_info



if __name__ == '__main__':
    class_labels = ['Standing',
                    'Sitting',
                    'Walking',
                    'Stand-to-walk',
                    'Stand-to-sit',
                    'Sit-to-stand',
                    'Walk-to-stand',
                    'Sit-to-walk',
                    'Walk-to-sit']

    x_train, y_train, x_test, y_test = load_data()

    clf = MLPClassifier(solver='adam',
                        alpha=1e-4,
                        hidden_layer_sizes=(32, 32, 16),
                        max_iter=2000,
                        random_state=1)

    clf.fit(x_train, y_train)
    print('MLP train accuracy: ', clf.score(x_train, y_train))
    print('MLP test accuracy: ', clf.score(x_test, y_test))


    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    #plt.figure(figsize = (10,7))
    sn.heatmap(cm, annot=True)
    plt.show()
