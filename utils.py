import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import matplotlib.pyplot as plt


def create_dataset(df, window_size=1):
    data = df.values
    x, y = [], []
    for i in range(len(df)-window_size-1):
        a = data[i:(i+window_size), :-1].reshape(1,-1)
        x.append(a)
        y.append(data[i + window_size, -1] - 1)
    x = np.array(x)
    y = np.array(y)
    return x.reshape(x.shape[0],-1), y


def load_data_sample():
    print('loading data...')
    df = pd.read_excel('data/Alwin_Round_1_raw_data_2019-10-21_11-2-5_sensor2.xlsx')
    df.dropna(axis=0, inplace=True)
    df_x = df.drop(['timestamp', 'label'], axis=1)
    print('df_x shape: ', df_x.shape)
    df_y = df['label']
    return df_x.values, df_y.values


def load_data(window_size=1):
    class_labels = ['Standing',
                'Sitting',
                'Walking',
                'Stand-to-walk',
                'Stand-to-sit',
                'Sit-to-stand',
                'Walk-to-stand',
                'Sit-to-walk',
                'Walk-to-sit']

    print('loading data...')
    '''
    df1 = pd.read_excel('data/Alwin_Round_1_raw_data_2019-10-21_11-2-5_sensor2.xlsx')
    df2 = pd.read_excel('data/Alwin_Round_2_raw_data_2019-10-21_11-9-2_sensor2.xlsx')
    df3 = pd.read_excel('data/Alwin_Round_3_raw_data_2019-10-21_11-16-23_sensor2.xlsx')
    df4 = pd.read_excel('data/Moon_Round_1_raw_data_2019-10-21_10-47-58_sensor2.xlsx')
    df5 = pd.read_excel('data/Moon_Round_2_raw_data_2019-10-21_10-54-30_sensor2.xlsx')
    df6 = pd.read_excel('data/Thu_Round_1_raw_data_2019-10-21_10-34-8_sensor2_Labeled.xlsx')
    df7 = pd.read_excel('data/Thu_Round_2_raw_data_2019-10-21_10-40-38_sensor2.xlsx')
    df = pd.concat([df1, df2, df3, df4, df5, df6, df7])
    df.dropna(axis=0, inplace=True)
    df.drop(['timestamp'], axis=1, inplace=True)
    df.to_csv('data/merged.csv')
    '''
    df = pd.read_csv('data/merged.csv')
    if window_size == 1:
        X = df.drop(['label'], axis=1).values
        y = df['label'].values - 1 # labels start at 0
    else:
        X, y = create_dataset(df, window_size=window_size)
    
    # Majority class classifier
    #acc = df_y.value_counts().values[0] / df_y.value_counts().values.sum()
    #print('Majority class classifier accuracy:', acc)
    #print('class distribution ', df_y.value_counts().values)
    #print('class distribution proportional ', df_y.value_counts().values / df_y.value_counts().values.sum())
    
    '''
    # plot bar chart of class distribution
    height = df_y.value_counts().values
    y_pos = np.arange(len(class_labels))
    plt.figure(figsize=(7, 7))
    plt.barh(y_pos, height, align='center', alpha=0.5)
    plt.yticks(y_pos, class_labels)
    plt.xlabel('Samples')
    plt.title('Class distribution')
    plt.show()
    '''

    return X, y


if __name__ == '__main__':
    load_data()