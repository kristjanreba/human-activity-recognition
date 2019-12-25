import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def load_data():
    df1 = pd.read_excel('data/Alwin_Round_1_raw_data_2019-10-21_11-2-5_sensor2.xlsx')
    df2 = pd.read_excel('data/Alwin_Round_2_raw_data_2019-10-21_11-9-2_sensor2.xlsx')
    df3 = pd.read_excel('data/Alwin_Round_3_raw_data_2019-10-21_11-16-23_sensor2.xlsx')
    df4 = pd.read_excel('data/Moon_Round_1_raw_data_2019-10-21_10-47-58_sensor2.xlsx')
    df5 = pd.read_excel('data/Moon_Round_2_raw_data_2019-10-21_10-54-30_sensor2.xlsx')
    df6 = pd.read_excel('data/Thu_Round_1_raw_data_2019-10-21_10-34-8_sensor2_Labeled.xlsx')
    df7 = pd.read_excel('data/Thu_Round_2_raw_data_2019-10-21_10-40-38_sensor2.xlsx')
    df = pd.concat([df1, df2, df3, df4, df5, df6, df7])
    df.dropna(axis=0, inplace=True)

    df_x = df.drop(['timestamp', 'label'], axis=1)
    df_y = df['label']
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)
    return x_train, y_train, x_test, y_test


def data_info():
    #print(df.info())
    #print(df.head())
    #print(df.describe())
    print('class distribution ', y_test.value_counts().values)
    print('class distribution proportional ', y_test.value_counts().values / y_test.value_counts().values.sum())
    

if __name__ == '__main__':
    pass