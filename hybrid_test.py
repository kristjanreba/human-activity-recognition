import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from utils import load_data_sample, load_data, data_info

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, TimeDistributed


def load_data_sample(num_classes, timesteps=1):
    print('loading data...')
    df = pd.read_excel('data/Alwin_Round_1_raw_data_2019-10-21_11-2-5_sensor2.xlsx')
    df.drop(['timestamp'], axis=1, inplace=True)
    df.dropna(axis=0, inplace=True)
    print(df.shape)
    print('creating dataset...')
    x, y = create_dataset(df, timesteps)
    print('x.shape =', x.shape)
    print('y.shape =', y.shape)
    y = keras.utils.to_categorical(y, num_classes=num_classes)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    return x_train, y_train, x_test, y_test


def load_data(num_classes, timesteps=1):
    print('loading data...')
    df1 = pd.read_excel('data/Alwin_Round_1_raw_data_2019-10-21_11-2-5_sensor2.xlsx')
    df2 = pd.read_excel('data/Alwin_Round_2_raw_data_2019-10-21_11-9-2_sensor2.xlsx')
    df3 = pd.read_excel('data/Alwin_Round_3_raw_data_2019-10-21_11-16-23_sensor2.xlsx')
    df4 = pd.read_excel('data/Moon_Round_1_raw_data_2019-10-21_10-47-58_sensor2.xlsx')
    df5 = pd.read_excel('data/Moon_Round_2_raw_data_2019-10-21_10-54-30_sensor2.xlsx')
    df6 = pd.read_excel('data/Thu_Round_1_raw_data_2019-10-21_10-34-8_sensor2_Labeled.xlsx')
    df7 = pd.read_excel('data/Thu_Round_2_raw_data_2019-10-21_10-40-38_sensor2.xlsx')
    df = pd.concat([df1, df2, df3, df4, df5, df6, df7])
    df.drop(['timestamp'], axis=1, inplace=True)
    df.dropna(axis=0, inplace=True)
    print(df.shape)
    print('creating dataset...')
    x, y = create_dataset(df, timesteps)
    print('x.shape =', x.shape)
    print('y.shape =', y.shape)
    y = keras.utils.to_categorical(y, num_classes=num_classes)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    return x_train, y_train, x_test, y_test


def create_dataset(df, timesteps=1):
    data = df.values
    x, y = [], []
    for i in range(len(df)-timesteps-1):
        a = data[i:(i+timesteps), :-2]
        x.append(a)
        y.append(data[i + timesteps, -1] - 1)
    return np.array(x), np.array(y).reshape((-1, 1))


def create_LSTM_model(data_dim, timesteps=1):
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, return_sequences=False)) # return a single vector of dimension 32
    model.add(Dense(9, activation='softmax'))
    model.summary()
    return model


if __name__ == '__main__':
    batch_size = 32
    epochs = 15
    timesteps = 32
    data_dim = 12
    num_classes = 9

    x_train, y_train, x_test, y_test = load_data(num_classes, timesteps)

    print('creating model...')
    model = create_LSTM_model(data_dim, timesteps)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('training...')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    
    print('making prediction...')
    y_pred_train = model.predict_classes(x_train)
    y_pred_test = model.predict_classes(x_test)

    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    print('LSTM train accuracy: ', accuracy_score(y_train, y_pred_train))
    print('LSTM test accuracy: ', accuracy_score(y_test, y_pred_test))
    
    cm = confusion_matrix(y_test, y_pred_test)
    sn.heatmap(cm, annot=True)
    plt.show()