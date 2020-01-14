import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from getCategories import Category


if __name__ == '__main__':
    mnist = np.load("../Dataset/DataSet.npy", allow_pickle=True)
    random.shuffle(mnist)

    x_train = np.array([mnist[i].features for i in range(0, int(len(mnist) * 0.8))])
    x_test = np.array([mnist[i].features for i in range(int(len(mnist) * 0.8), len(mnist))])
    y_train = np.array([mnist[i].label for i in range(0, int(len(mnist) * 0.8))])
    y_test = np.array([mnist[i].label for i in range(int(len(mnist) * 0.8), len(mnist))])
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)

    model = Sequential()
    model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=0.005, decay=1e-6)

    # Compile model
    model.compile(loss='kullback_leibler_divergence',optimizer=opt, metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))