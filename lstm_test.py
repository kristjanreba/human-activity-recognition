import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import RandomOverSampler, SMOTE

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, TimeDistributed


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def CV(x, y, timesteps):
    batch_size = 32
    epochs = 3
    data_dim = 12
    num_classes = 9

    accs = []
    kf = KFold(n_splits=5, random_state=1, shuffle=True)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #print('creating model...')
        model = create_LSTM_model(data_dim, timesteps)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #print('training...')
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
        
        #print('making prediction...')
        y_pred_train = model.predict_classes(x_train)
        y_pred_test = model.predict_classes(x_test)

        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        print('LSTM train accuracy: ', train_acc)
        print('LSTM test accuracy: ', test_acc)
        accs.append(test_acc)

    std = np.std(accs)
    mean = np.mean(accs)
    print('LSTM acc = {} +/- {}'.format(mean, std))


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
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    #return x_train, y_train, x_test, y_test
    return x, y


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
    x, y = create_dataset(df, timesteps)
    #x = (x - x.mean()) / x.std() # standard normalization
    x, y = unison_shuffled_copies(x, y)
    y = keras.utils.to_categorical(y, num_classes=num_classes)
    return x, y


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
    #model.summary()
    return model


if __name__ == '__main__':
    '''
    num_classes = 9
    batch_size = 32
    epochs = 4
    data_dim = 12
    ts = [512]

    accs = []
    for timesteps in ts:
        print('Current timesteps: ', timesteps)
        x, y = load_data(num_classes, timesteps)
        #CV(x, y, timesteps)

        # split the data into train and test sets
        train_size = int(0.7 * x.shape[0])
        x_train = x[:train_size,:,:]
        y_train = y[:train_size,:]
        x_test = x[train_size:,:,:]
        y_test = y[train_size:,:]

        # oversample the train set
        #sm = SMOTE(random_state=0)
        #x, y = sm.fit_resample(x, y)

        # create and train the model
        model = create_LSTM_model(data_dim, timesteps)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

        # evaluate the model
        y_pred_train = model.predict_classes(x_train)
        y_pred_test = model.predict_classes(x_test)

        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)

        print('LSTM train accuracy: ', accuracy_score(y_train, y_pred_train))
        print('LSTM test accuracy: ', accuracy_score(y_test, y_pred_test))
        accs.append(accuracy_score(y_test, y_pred_test))

    print('accs = ', accs)
    '''

    '''
    batch_size = 32
    epochs = 4
    timesteps = 256
    data_dim = 12
    num_classes = 9

    x, y = load_data(num_classes, timesteps=timesteps)

    train_size = int(0.7 * x.shape[0])
    x_train = x[:train_size,:,:]
    y_train = y[:train_size,:]
    x_test = x[train_size:,:,:]
    y_test = y[train_size:,:]

    # oversample the dataset
    sm = SMOTE(random_state=0)
    x_train_orig_shape = x_train.shape
    y_train_orig_shape = y_train.shape
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_train, y_train = sm.fit_resample(x_train, y_train)
    x_train = np.reshape(x_train, (-1, x_train_orig_shape[1], x_train_orig_shape[2]))
    
    model = create_LSTM_model(data_dim, timesteps)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    y_pred_train = model.predict_classes(x_train)
    y_pred_test = model.predict_classes(x_test)

    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    print('LSTM train accuracy: ', accuracy_score(y_train, y_pred_train))
    print('LSTM test accuracy: ', accuracy_score(y_test, y_pred_test))
    '''


    '''
    # plot confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    sn.heatmap(cm, annot=True)
    plt.show()
    '''


    # plot bar chart of class distribution
    ts = [2, 4, 8, 16, 32, 64, 128, 256]
    accs = [0.9215973442686975, 0.9248519169433054, 0.9327886993881005, 0.9432328624438513, 0.9607383533548198, 0.9648625765272894, 0.9678375912408759, 0.9699836867862969]
    height = accs
    y_pos = np.arange(len(ts))
    plt.barh(y_pos, height, align='center', alpha=0.5)
    plt.yticks(y_pos, ts)
    plt.ylabel('Window size')
    plt.xlabel('Accuracy')
    plt.xlim((0.9, 1.0))
    plt.title('Increasing window size')
    plt.show()
