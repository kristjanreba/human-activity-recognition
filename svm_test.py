import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def load_data():
    df = pd.read_excel('data/Alwin_Round_1_raw_data_2019-10-21_11-2-5_sensor2.xlsx')
    #print(df.info())
    #print(df.head())
    #print(df.describe())
    df_x = df.drop(['timestamp', 'label'], axis=1)
    df_y = df['label']
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)
    return x_train, y_train, x_test, y_test



if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    '''
    Cs = np.logspace(-6, 3, 10)
    parameters = [{'kernel': ['rbf'], 'C': Cs},
                {'kernel': ['linear'], 'C': Cs}]
    '''

    Cs = [1000]
    parameters = [{'kernel': ['rbf'], 'C': Cs}]

    svc = SVC(random_state = 12, gamma='scale')
    clf = GridSearchCV(estimator = svc, param_grid = parameters, cv = 5, n_jobs = -1)
    clf.fit(x_train.values, y_train.values.flatten())
    #print(clf.best_params_)
    #print(clf.best_score_)
    print('SVC accuracy: ', clf.score(x_test, y_test))

    # Majority class classifier
    acc = y_test.value_counts().values[0] / y_test.value_counts().values.sum()
    print('Majority class classifier accuracy:', acc)