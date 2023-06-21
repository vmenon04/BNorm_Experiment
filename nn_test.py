import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



import tensorflow as tf

import numpy as np
import h5py

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

import pandas as pd

import time
import sys

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

layer_list = [10000,5000,1000,500,200,20] # hidden layer widths

num_layers = int(sys.argv[1])
activation_type = sys.argv[2]

assert num_layers <= len(layer_list), "you are asking for too many layers"

# print("Layers:"+str(num_layers))

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

train_x, train_y, test_x, test_y, classes = load_data()

x = train_x
y = train_y

def create_raw_model(learning_rate = 0.0075):
    # Neural network structure without Batch Normalization
    model_raw = keras.Sequential()
    model_raw.add(layers.Flatten(input_shape=(64,64,3)))
    for i in range(num_layers):
        model_raw.add(layers.Dense(units=layer_list[i], name='raw_dense{0}'.format(i+1), activation=activation_type))
    model_raw.add(layers.Dense(units=1, name='raw_output', activation='sigmoid'))

    # compile the raw model
    model_raw.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['acc']
    )

    return model_raw

def create_bnorm_model(learning_rate = 0.0075):
    # Neural network structure for Batch Normalized Model
    model_bnorm = keras.Sequential()
    model_bnorm.add(layers.Flatten(input_shape=(64,64,3)))
    for i in range(num_layers):
        model_raw.add(layers.Dense(units=layer_list[i], name='bnorm_dense{0}'.format(i+1), activation=activation_type))
        model_bnorm.add(layers.BatchNormalization(name='bnorm_bnorm{0}'.format(i+1)))
    model_bnorm.add(layers.Dense(units=1, name='bnorm_output', activation='sigmoid'))

    # compile the bnorm model
    model_bnorm.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    return model_bnorm

iterations = 10
epochs = 1500
n_splits = 5
kFold = KFold(n_splits=n_splits, shuffle=True)

data = pd.DataFrame(columns=['Iteration', 'Fold #', 'Raw Loss', 'BNorm Loss', 'Raw Accuracy', 'BNorm Accuracy', 'Raw Precision', 'BNorm Precision', 'Raw Recall', 'BNorm Recall'])
loss_acc_df = pd.DataFrame(columns=['Iteration', 'Fold #', 'Raw Loss', 'BNorm Loss', 'Raw Accuracy', 'BNorm Accuracy'])
prec_rec_df = pd.DataFrame(columns=['Iteration', 'Fold #', 'Sample ID', 'Actual', 'Raw', 'BNorm'])
time_df = pd.DataFrame(columns=['Iteration', 'Fold #', 'Raw Time', 'BNorm Time'])


loss_raw = np.zeros(n_splits)
loss_bnorm = np.zeros(n_splits)
acc_raw = np.zeros(n_splits)
acc_bnorm = np.zeros(n_splits)
precision_raw = np.zeros(n_splits)
precision_bnorm = np.zeros(n_splits)
recall_raw = np.zeros(n_splits)
recall_bnorm = np.zeros(n_splits)

for iteration in range(iterations):
    idx = 0
    # print("Iteration: " + str(iteration+1))
    for train, test in kFold.split(x, y.T):
        # print("Split #" + str(idx+1))

        # Train and Evaluate Raw Model

        raw_start_time = time.time()
        model_raw = create_raw_model()
        history_raw = model_raw.fit(x[train], y.T[train], epochs = epochs, batch_size = 209, verbose=0,)
        loss_raw[idx] = history_raw.history['loss'][-1]
        raw_y_pred1 = model_raw.predict(x[test], verbose=0,)
        raw_y_pred = np.where(raw_y_pred1 > 0.5, 1, 0)
        precision_raw[idx] = precision_score(y.T[test], raw_y_pred)
        recall_raw[idx] = recall_score(y.T[test], raw_y_pred)
        _, acc_raw[idx] = model_raw.evaluate(x[test], y.T[test], verbose=0,)
        raw_end_time = time.time()
        raw_total_time = raw_end_time - raw_start_time
        keras.backend.clear_session()

        # Train and Evaluate BNorm Model

        bnorm_start_time = time.time()
        model_bnorm = create_bnorm_model()
        history_bnorm = model_bnorm.fit(x[train], y.T[train], epochs = epochs, batch_size = 209, verbose=0,)
        loss_bnorm[idx] = history_bnorm.history['loss'][-1]
        bnorm_y_pred1 = model_bnorm.predict(x[test], verbose=0,)
        bnorm_y_pred = np.where(bnorm_y_pred1 > 0.5, 1, 0)
        precision_bnorm[idx] = precision_score(y.T[test], bnorm_y_pred)
        recall_bnorm[idx] = recall_score(y.T[test], bnorm_y_pred)
        _, acc_bnorm[idx] = model_bnorm.evaluate(x[test], y.T[test], verbose=0,)
        bnorm_end_time = time.time()
        bnorm_total_time = bnorm_end_time - bnorm_start_time
        keras.backend.clear_session()

        # Write Data
        for sub_idx in range(len(test)):
            expected_output = np.squeeze(y.T[test][sub_idx])
            raw_output = np.squeeze(raw_y_pred[sub_idx])
            bnorm_output = np.squeeze(bnorm_y_pred[sub_idx])
            new_row  = {'Iteration': int(iteration+1), 'Fold #': int(idx+1), 'Sample ID': test[sub_idx], 'Actual': expected_output, 'Raw': raw_output, 'BNorm': bnorm_output}
            prec_rec_df = prec_rec_df.append(new_row, ignore_index = True)


        # iteration, idx+1, loss_raw, loss_bnorm, acc_raw, acc_bnorm
        new_row  = {'Iteration': int(iteration+1), 'Fold #': int(idx+1), 'Raw Loss': loss_raw[idx], 'BNorm Loss': loss_bnorm[idx], 'Raw Accuracy': acc_raw[idx], 'BNorm Accuracy': acc_bnorm[idx]}
        loss_acc_df = loss_acc_df.append(new_row, ignore_index = True)

        # iteration, idx+1, loss_raw, loss_bnorm, acc_raw, acc_bnorm, precision_raw, precision_bnorm, recall_raw, recall_bnorm
        new_row  = {'Iteration': int(iteration+1), 'Fold #': int(idx+1), 'Raw Loss': loss_raw[idx], 'BNorm Loss': loss_bnorm[idx], 'Raw Accuracy': acc_raw[idx], 'BNorm Accuracy': acc_bnorm[idx], 'Raw Precision': precision_raw[idx], 'BNorm Precision': precision_bnorm[idx], 'Raw Recall': recall_raw[idx], 'BNorm Recall': recall_bnorm[idx]}
        data = data.append(new_row, ignore_index = True)

        new_row = {'Iteration': int(iteration+1), 'Fold #': int(idx+1), 'Raw Time': raw_total_time, 'BNorm Time': bnorm_total_time}
        time_df = time_df.append(new_row, ignore_index = True)

        idx += 1


    # print("Raw Losses: " + str(loss_raw))
    # print("Raw Losses Mean: " + str(loss_raw.mean()))
    # print("BNorm Losses: " + str(loss_bnorm))
    # print("BNorm Losses Mean: " + str(loss_bnorm.mean()))

    # print("Raw Accuracy: " + str(acc_raw))
    # print("Raw Accuracy Mean: " + str(acc_raw.mean()))
    # print("BNorm Accuracy: " + str(acc_bnorm))
    # print("BNorm Accuracy Mean: " + str(acc_bnorm.mean()))

    # print("Raw Precision: " + str(precision_raw))
    # print("Raw Precision Mean: " + str(precision_raw.mean()))
    # print("BNorm Precision: " + str(precision_bnorm))
    # print("BNorm Precision Mean: " + str(precision_bnorm.mean()))

    # print("Raw Recall: " + str(recall_raw))
    # print("Raw Recall Mean: " + str(recall_raw.mean()))
    # print("BNorm Recall: " + str(recall_bnorm))
    # print("BNorm Recall Mean: " + str(recall_bnorm.mean()))

t1 = time.time()
total = t1-t0

a = prec_rec_df["Actual"].astype(int)
b = prec_rec_df["Raw"].astype(int)
c = prec_rec_df["BNorm"].astype(int)

raw_times_mean = np.mean(time_df["Raw Time"])
bnorm_times_mean = np.mean(time_df["BNorm Time"])

raw_precision_score = precision_score(a, b)
raw_recall_score = recall_score(a, b)
bnorm_precision_score = precision_score(a, c)
bnorm_recall_score = recall_score(a, c)

print(str(num_layers) + "," + str(raw_precision_score) + "," + str(bnorm_precision_score) + "," + str(raw_recall_score) + "," + str(bnorm_recall_score) + "," +  str(raw_times_mean) + "," + str(bnorm_times_mean))

# print("Raw:" + str(precision_score(a, b)) + "," + str(recall_score(a, b)))
# print("BNorm:" + str(precision_score(a, c)) + "," + str(recall_score(a, c)))

# print("---------")
# print("Time:" + str(total))

