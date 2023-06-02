import tensorflow as tf

import numpy as np
import h5py

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import StratifiedKFold

import pandas as pd

import time

t0 = time.time()

def load_data():
    train_dataset = h5py.File('./train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('./test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
  
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
  
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

x = train_x_orig
y = train_y

# Neural network structure without Batch Normalization
model_raw = keras.Sequential()
model_raw.add(layers.Flatten(input_shape=(64,64,3)))
model_raw.add(layers.Dense(units=20, input_shape=(1,12288), activation='relu'))
model_raw.add(layers.Dense(units=7, input_shape=(1,20), activation='relu'))
model_raw.add(layers.Dense(units=5, input_shape=(1,7), activation='relu'))
model_raw.add(layers.Dense(units=1, input_shape=(1,5), activation='sigmoid'))

# Neural network structure for Batch Normalized Model
model_bnorm = keras.Sequential()
model_bnorm.add(layers.Flatten(input_shape=(64,64,3)))
model_bnorm.add(layers.Dense(units=20, input_shape=(1,12288), activation='relu'))
model_bnorm.add(layers.BatchNormalization())
model_bnorm.add(layers.Dense(units=7, input_shape=(1,20), activation='relu'))
model_bnorm.add(layers.BatchNormalization())
model_bnorm.add(layers.Dense(units=5, input_shape=(1,7), activation='relu'))
model_bnorm.add(layers.BatchNormalization())
model_bnorm.add(layers.Dense(units=1, input_shape=(1,5), activation='sigmoid'))


learning_rate = 0.0075

# compile the raw model
model_raw.compile(
    optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# compile the bnorm model
model_bnorm.compile(
    optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)


# Do CrossValidation
df = pd.DataFrame(columns=['Iteration', 'Fold #', 'Raw Loss', 'BNorm Loss', 'Raw Accuracy', "BNorm Accuracy"])

iterations = 1
n_splits = 5
epochs = 2500
kFold = StratifiedKFold(n_splits=n_splits)
loss_raw = np.zeros(n_splits)
loss_bnorm = np.zeros(n_splits)
acc_raw = np.zeros(n_splits)
acc_bnorm = np.zeros(n_splits)

for iteration in range(iterations):
    idx = 0
    for train, test in kFold.split(x, y.T):
        history_raw = model_raw.fit(x[train], y.T[train], epochs = epochs, batch_size = 209, validation_split = 0.2, verbose=0,)
        history_bnorm = model_bnorm.fit(x[train], y.T[train], epochs = epochs, batch_size = 209, validation_split = 0.2, verbose=0,)

        loss_raw[idx] = history_raw.history['loss'][-1]
        loss_bnorm[idx] = history_bnorm.history['loss'][-1]

        _, raw_test_acc = model_raw.evaluate(x[test], y.T[test])
        _, bnorm_test_acc = model_bnorm.evaluate(x[test], y.T[test])

        acc_raw[idx] = raw_test_acc
        acc_bnorm[idx] = bnorm_test_acc
        
        # iteration, idx+1, loss_raw, loss_bnorm, acc_raw, acc_bnorm
        new_row  = {'Iteration': int(iteration+1), 'Fold #': int(idx+1), 'Raw Loss': loss_raw[idx], 'BNorm Loss': loss_bnorm[idx], 'Raw Accuracy': raw_test_acc, 'BNorm Accuracy': bnorm_test_acc}
        df = df.append(new_row, ignore_index = True)
        
        idx += 1

    print("Raw Losses: " + str(loss_raw))
    print("Raw Losses Mean: " + str(loss_raw.mean()))
    print("BNorm Losses: " + str(loss_bnorm))
    print("BNorm Losses Mean: " + str(loss_bnorm.mean()))

    print("Raw Accuracy: " + str(acc_raw))
    print("Raw Accuracy Mean: " + str(acc_raw.mean()))
    print("BNorm Accuracy: " + str(acc_bnorm))
    print("BNorm Accuracy Mean: " + str(acc_bnorm.mean()))


t1 = time.time()
total = t1-t0

print("---------")
print("Time: " + str(total))

df.to_csv('./data.csv', index=False)
