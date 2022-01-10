# RNN Classifier
import numpy as np
import pandas as pd
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras import models
from tensorflow.keras import callbacks
from manual_early_stop import TerminateOnBaseline

tf.keras.backend.clear_session()
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K


# ======================================================================================================
def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


def LSTMFCN_classifier(save_path, filename, x_train, y_train, x_val, y_val, x_test, y_test, nb_classes, run, verbose=False, min_exp_val_loss=0.005):
    np.random.seed()
    batch_size = 50
    nb_epochs = 500
    input_layer = Input(shape=x_train.shape[1:])

    x = layers.LSTM(16)(input_layer)
    x = layers.Dropout(0.8)(x)
    y = layers.Permute((2, 1))(input_layer)
    y = layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.GlobalAveragePooling1D()(y)
    x = layers.concatenate([x, y])
    output_layer = layers.Dense(nb_classes, activation='softmax')(x)
    model = models.Model(input_layer, output_layer)

    file_path = save_path + 'models/' + filename + '_LSTMFCN_best_model_run_' + str(run) + '.hdf5'
    mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))
    model.compile(optimizer=keras.optimizers.Adam(), loss='mse')
    model_checkpoint = callbacks.ModelCheckpoint(filepath=file_path, verbose=1, save_best_only=True)

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_main_output_loss', factor=0.92, patience=5, min_lr=0.0001)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=min_exp_val_loss, patience=15)
    baseline_stop = TerminateOnBaseline(monitor='val_acc', baseline=1.0, patience=15)

    start_time = time.time()
    hist = model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                     validation_data=(x_val, y_val), verbose=2,
                     callbacks=[
                         early_stop, baseline_stop,
                         reduce_lr,
                         model_checkpoint])
    duration = time.time() - start_time
    print("Elapsed Training Time: %f" % (duration))
    model = models.load_model(file_path)
    y_val_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)
    training_itrs = len(hist.history['loss'])
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(save_path + 'models/' + filename + '_LSTMFCN_history_run_' + str(run) + '.csv', index=False)
    keras.backend.clear_session()

    if verbose == 1:
        val_perf = mean_absolute_error(y_val, y_val_pred), sqrt(mean_squared_error(y_val, y_val_pred))
        test_perf = mean_absolute_error(y_test, y_test_pred), sqrt(mean_squared_error(y_test, y_test_pred))
        print(f'MAE= {abs(val_perf[0] - test_perf[0]) < 0.01}, val: {val_perf[0]}, test: {test_perf[0]}')
        print(f'RMSE= {abs(val_perf[1] - test_perf[1]) < 0.01}, val: {val_perf[1]}, test: {test_perf[1]}')

    return y_train_pred, y_val_pred, y_test_pred, duration, training_itrs
