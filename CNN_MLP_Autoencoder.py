# CNN-MLP-Autoencoder Classifier
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
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error,mean_absolute_error
from math import sqrt
tf.keras.backend.clear_session()


# ======================================================================================================

def CNN_MLP_Autoencoder(save_path, filename, x_train, y_train, x_val, y_val, x_test, y_test, nb_classes, run, verbose=False, min_exp_val_loss=0.005):
    np.random.seed()
    batch_size = 50
    nb_epochs = 500
    n_channels = x_train.shape[-1]
    laten_dim = 2
    # Encoder Model
    input_layer = Input(shape=x_train.shape[1:])
    x = layers.Conv1D(16, 5, 1, padding='same', activation='relu')(input_layer)
    # enabled_crop = False
    # if x.shape[1] % 2 != 0:
    #     x = layers.ZeroPadding1D((1, 0))(x)
    #     enabled_crop = True
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(2, 5, 1, padding='same', activation='relu')(x)

    # shape_before_flattening = K.int_shape(x)
    # x = layers.Flatten()(x)
    # encoder_output = layers.Dense(laten_dim, activation='relu')(x)
    # # encoder = models.Model(inputs = input_layer, outputs = encoder_output, name='encoder')
    # # Decoder Model
    # x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(encoder_output)
    # x = layers.Reshape(shape_before_flattening[1:])(x)

    x = layers.Conv1D(2, 5, 1, padding='same', activation='relu')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(16, 5, 1, padding='same', activation='relu')(x)
    output_layer = layers.Conv1D(n_channels, 5, 1, padding='same', name='decoder_output', activation='relu')(x)
    # if enabled_crop:
    #     output_layer = layers.Cropping1D(cropping=(1, 0))(output_layer)  # this is the added step
    #
    # if input_layer[1].shape != output_layer[1].shape:
    #     output_layer = layers.Cropping1D(cropping=(1, 1))(output_layer)  # this is the added step

    # Autoencoder Model
    autoencoder = models.Model(input_layer, output_layer, name='autoencoder_cnn')
    autoencoder.compile(loss='mse', optimizer=keras.optimizers.Adam())
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.92, patience=3)
    early_stop = callbacks.EarlyStopping(monitor='loss', min_delta=1e-3, patience=15)
    file_path = save_path + 'models/' + filename + '_Autoencoder(CNN)_best_model_run_' + str(run) + '.hdf5'
    model_checkpoint = callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)

    start_time = time.time()
    autoencoder.fit(x_train, x_train, validation_data=(x_val, x_val), batch_size=128, epochs=50,
                    callbacks=[early_stop, reduce_lr, model_checkpoint])

    autoencoder = models.load_model(file_path)
    autoencoder.summary()
    # if enabled_crop:
    #     encoder = models.Model(inputs=autoencoder.inputs, outputs=autoencoder.layers[5].output)
    # else:
    encoder = models.Model(inputs=autoencoder.inputs, outputs=autoencoder.layers[3].output)
    # Encode  sets
    x_train_encode = encoder.predict(x_train)
    x_val_encode = encoder.predict(x_val)
    x_test_encode = encoder.predict(x_test)

    # Train MLP on encoded inputs
    input_layer = Input(shape=x_train_encode.shape[1:])
    x = layers.Flatten()(input_layer)
    x = layers.Dense(100, activation='relu')(x)
    output_layer = layers.Dense(nb_classes)(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam())

    model.summary()
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=50)
    baseline_stop = TerminateOnBaseline(monitor='val_acc', baseline=1.0, patience=15)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.92, patience=3)
    file_path = save_path + 'models/' + filename + '_CNN_MLP_Autoencoder_best_model_run_' + str(run) + '.hdf5'
    model_checkpoint = callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
    mini_batch_size = int(min(x_train_encode.shape[0] / 10, batch_size))

    hist = model.fit(x_train_encode, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                     validation_data=(x_val_encode, y_val), verbose=2,
                     callbacks=[early_stop, baseline_stop, reduce_lr, model_checkpoint])
    duration = time.time() - start_time
    print("Elapsed Training Time: %f" % (duration))
    model = models.load_model(file_path)
    y_val_pred = model.predict(x_val_encode)
    y_train_pred = model.predict(x_train_encode)
    y_test_pred = model.predict(x_test_encode)
    training_itrs = len(hist.history['loss'])
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(save_path + 'models/' + filename + '_CNN_MLP_Autoencoder_history_run_' + str(run) + '.csv',
                   index=False)
    keras.backend.clear_session()

    if verbose ==1:
        val_perf = mean_absolute_error(y_val, y_val_pred), sqrt(mean_squared_error(y_val, y_val_pred))
        test_perf = mean_absolute_error(y_test, y_test_pred), sqrt(mean_squared_error(y_test, y_test_pred))
        print(f'MAE= {abs(val_perf[0] - test_perf[0]) < 0.01}, val: {val_perf[0]}, test: {test_perf[0]}')
        print(f'RMSE= {abs(val_perf[1] - test_perf[1]) < 0.01}, val: {val_perf[1]}, test: {test_perf[1]}')

    return y_train_pred, y_val_pred, y_test_pred, duration, training_itrs
