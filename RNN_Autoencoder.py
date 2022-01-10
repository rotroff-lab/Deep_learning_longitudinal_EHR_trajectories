# import numpy as np
# import pandas as pd
# import time
#
# import tensorflow as tf
# from matplotlib import pyplot
# from tensorflow import keras
# from tensorflow.keras import Input
# from tensorflow.keras.layers import LSTM, Dense, RepeatVector,TimeDistributed
# from tensorflow.keras.utils import plot_model
# from tensorflow.keras import regularizers
# from tensorflow.keras import models, Sequential
# from tensorflow.keras import callbacks
# from manual_early_stop import TerminateOnBaseline
# from tensorflow.keras import backend as K
# tf.keras.backend.clear_session()
#
# def RNN_Autoencoder(save_path, filename, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, nb_classes, run, verbose=False, min_exp_val_loss=0.005):
#         np.random.seed()
#         serie_size =  X_train.shape[1] # 12
#         n_features =  X_train.shape[2] # 1
#
#         epochs = 20
#         batch = 128
#         lr = 0.0001
#
#         encoder_decoder = Sequential()
#         encoder_decoder.add(LSTM(serie_size, activation='relu', input_shape=(serie_size, n_features), return_sequences=True))
#         encoder_decoder.add(LSTM(6, activation='relu', return_sequences=True))
#         encoder_decoder.add(LSTM(1, activation='relu'))
#         encoder_decoder.add(RepeatVector(serie_size))
#         encoder_decoder.add(LSTM(serie_size, activation='relu', return_sequences=True))
#         encoder_decoder.add(LSTM(6, activation='relu', return_sequences=True))
#         encoder_decoder.add(TimeDistributed(Dense(1)))
#         encoder_decoder.summary()
#
#         file_path = save_path + 'models/' + filename + '_Autoencoder(RNN)_best_model_run_' + str(run) + '.hdf5'
#         model_checkpoint = callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)
#
#         adam = keras.optimizers.Adam(lr)
#         encoder_decoder.compile(loss='mse', optimizer=adam)
#         encoder_decoder_history = encoder_decoder.fit(X_train, X_train,
#                                                       batch_size=batch,
#                                                       epochs=epochs,
#                                                       verbose=2, callbacks=[model_checkpoint])
#         encoder_decoder = models.load_model(file_path)
#         encoder = models.Model(inputs=encoder_decoder.inputs, outputs=encoder_decoder.layers[2].output)
#
#         X_train_encoded = encoder.predict(X_train)
#         X_valid_encoded = encoder.predict(X_valid)
#         X_test_encoded = encoder.predict(X_test)
#
#         last_data = serie_size - 1
#         X_train_encoded = np.hstack((X_train[:,last_data,:],X_train_encoded))
#         X_valid_encoded = np.hstack((X_valid[:,last_data,:],X_valid_encoded))
#         X_test_encoded = np.hstack((X_test[:,last_data,:],X_test_encoded))
#
#
#
#         mlp_model = Sequential()
#         mlp_model.add(Dense(10, kernel_initializer='glorot_normal', activation='relu', input_dim=X_train_encoded.shape[1]))
#         mlp_model.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
#         mlp_model.add(Dense(nb_classes))
#         mlp_model.summary()
#         adam = keras.optimizers.Adam(lr)
#         mlp_model.compile(loss='mse', optimizer=adam)
#         mlp_history = mlp_model.fit(X_train_encoded, Y_train, epochs=epochs, batch_size=batch,
#                                     validation_data=(X_valid_encoded, Y_valid), verbose=2)
#
#         y_val_pred = mlp_model.predict(X_valid_encoded)
#         y_train_pred = mlp_model.predict(X_train_encoded)
#         y_test_pred = mlp_model.predict(X_test_encoded)
#         training_itrs = len(mlp_history.history['loss'])
#         hist_df = pd.DataFrame(mlp_history.history)
#         hist_df.to_csv(save_path + 'models/' + filename + '_RNN_Autoencoder_history_run_' + str(run) + '.csv', index=False)
#         keras.backend.clear_session()
#         return y_train_pred, y_val_pred, y_test_pred, 0, training_itrs


# Fully CNN Classifier
import numpy as np
import pandas as pd
import time

import tensorflow as tf
from matplotlib import pyplot
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras import models
from tensorflow.keras import callbacks
from manual_early_stop import TerminateOnBaseline
from tensorflow.keras import backend as K
tf.keras.backend.clear_session()
from sklearn.metrics import mean_squared_error,mean_absolute_error
from math import sqrt
from tensorflow.keras.optimizers import RMSprop,Adam,SGD
#======================================================================================================

def RNN_Autoencoder(save_path, filename, x_train, y_train, x_val, y_val, x_test, y_test, nb_classes, run, verbose=False, min_exp_val_loss=0.005):
        np.random.seed()
        batch_size = 50
        nb_epochs = 500
        n_channels = x_train.shape[-1]
        timesteps = x_train.shape[1]

        # AutoEncoder Model
        input_layer = Input(shape=x_train.shape[1:])
        x = layers.LSTM(16,activation='relu', return_sequences=True)(input_layer)
        encoder_output = layers.LSTM(2,activation='relu',)(x)
        encoder = models.Model(input_layer, encoder_output, name='encoder')
        x = layers.RepeatVector(timesteps)(encoder_output)
        x = layers.LSTM(16, activation='relu', return_sequences=True)(x)
        output_layer = layers.TimeDistributed(layers.Dense(n_channels))(x)
        autoencoder = models.Model(input_layer, output_layer, name='autoencoder_cnn')
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.92, patience=3)
        early_stop = callbacks.EarlyStopping(monitor='loss', min_delta=1e-3, patience=15)
        file_path = save_path + 'models/' + filename + '_Autoencoder(RNN)_best_model_run_' + str(run) + '.hdf5'
        model_checkpoint = callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
        autoencoder.compile(loss='mse', optimizer=keras.optimizers.Adam())

        start_time = time.time()
        autoencoder.fit(x_train, x_train, validation_data=(x_val, x_val), batch_size=128, epochs=50,
                        callbacks=[early_stop, reduce_lr,model_checkpoint])

        autoencoder = models.load_model(file_path)
        autoencoder.summary()
        encoder = models.Model(inputs=autoencoder.inputs, outputs=autoencoder.layers[2].output)


        # Encode Training Data
        x_train_encoded = encoder.predict(x_train)
        x_val_encoded = encoder.predict(x_val)
        x_test_encoded = encoder.predict(x_test)

        # Train MLP on encoded inputs
        input_layer = Input(shape=x_train_encoded.shape[1:])
        x = layers.Flatten()(input_layer)
        x = layers.Dense(100, activation='relu')(x)
        output_layer = layers.Dense(nb_classes)(x)
        model = models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mse', optimizer=keras.optimizers.Adam())
        early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=50)
        baseline_stop = TerminateOnBaseline(monitor='val_acc', baseline=1.0, patience=15)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.92, patience=3)
        file_path = save_path + 'models/' + filename + '_RNN_Autoencoder_best_model_run_' + str(run) + '.hdf5'
        model_checkpoint = callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
        mini_batch_size = int(min(x_train_encoded.shape[0] / 10, batch_size))

        hist = model.fit(x_train_encoded, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                         validation_data=(x_val_encoded, y_val),
                         callbacks=[early_stop, baseline_stop, reduce_lr, model_checkpoint])
        duration = time.time() - start_time
        print("Elapsed Training Time: %f" % (duration))
        model = models.load_model(file_path)
        model.summary()
        y_val_pred = model.predict(x_val_encoded)
        y_train_pred = model.predict(x_train_encoded)
        y_test_pred = model.predict(x_test_encoded)
        training_itrs = len(hist.history['loss'])
        hist_df = pd.DataFrame(hist.history)
        hist_df.to_csv(save_path + 'models/' + filename + '_RNN_Autoencoder_history_run_' + str(run) + '.csv', index=False)
        keras.backend.clear_session()


        # pyplot.plot(hist.history['loss'])
        # pyplot.plot(hist.history['val_loss'])
        # pyplot.title('model train vs validation loss')
        # pyplot.ylabel('loss')
        # pyplot.xlabel('epoch')
        # pyplot.legend(['train', 'validation'], loc='upper right')
        # pyplot.show()

        return y_train_pred, y_val_pred, y_test_pred, duration, training_itrs


# # RNN-Autoencoder Classifier
# from keras.layers import Dense, Dropout, BatchNormalization, RepeatVector, LSTM
# from keras.layers import *
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from math import sqrt
# import tensorflow as tf
# from keras.models import Sequential
# import os
# os.environ['PYTHONHASHSEED'] = '0'
# import warnings
#
# warnings.simplefilter("ignore", DeprecationWarning)
# # ======================================================================================================
#
#
#
# def RNN_Autoencoder(save_path, filename, x_train, y_train, x_val, y_val, x_test, y_test, nb_classes, run, verbose=False, min_exp_val_loss=0.005):
#
#     # train AE
#     batch_size = 50
#     mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))
#     timesteps = x_train.shape[1]
#     input_dim = x_train.shape[2]
#     AE = Sequential()
#     AE.add(LSTM(32, batch_input_shape=(mini_batch_size, timesteps, input_dim), stateful=False))
#     AE.add(RepeatVector(timesteps))
#     AE.add(LSTM(32, stateful=False, return_sequences=True))
#
#     AE.compile(loss='mean_squared_error', optimizer='Adam')
#
#     AE.fit(x_train, x_train,
#            epochs=500,
#            batch_size= mini_batch_size,
#            shuffle=True,
#            verbose=1
#            )
#     AE.summary()
#
#     trained_encoder = AE.layers[0]
#     weights = AE.layers[0].get_weights()
#
#     # Fine-turning
#     print('\nFine-turning')
#     print('============')
#
#     # build finetuning model
#
#     model = Sequential()
#     model.add(trained_encoder)
#     model.layers[-1].set_weights(weights)
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(nb_classes, activation='softmax'))
#
#     model.compile(loss='mean_squared_error', optimizer='Adam')
#
#     model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=100, batch_size=mini_batch_size, verbose=1, shuffle=True)
#     model.summary()
#
#     # redefine the model in order to test with one sample at a time (batch_size = 1)
#     new_model = Sequential()
#     new_model.add(LSTM(32, batch_input_shape=(1, timesteps, input_dim), stateful=False))
#     new_model.add(Dense(100, activation='relu'))
#     new_model.add(Dense(nb_classes, activation='softmax'))
#
#     # copy weights
#     old_weights = model.get_weights()
#     new_model.set_weights(old_weights)
#
#     # forecast the valid data
#     print('Forecasting valid Data')
#     y_test_pred = list()
#     for i in range(len(x_test)):
#         # make one-step forecast
#         X = x_test[i]
#         y = y_test[i]
#         X = X.reshape(1, X.shape[0], X.shape[1])
#         yhat = new_model.predict(X, batch_size=1)
#         y_test_pred.append(yhat)
#
#     # y_test_pred = new_model.predict(x_test)
#     # y_train_pred = new_model.predict(x_train)
#     # y_val_pred = new_model.predict(x_val)
#
#     # # report performance using RMSE
#     # rmse_valid = sqrt(mean_squared_error(raw_values[:, 0][len(train_scaled2) + 1:len(valid) + len(train_scaled2) + 1], predictions_valid))
#     # print('valid RMSE: %.5f' % rmse_valid)
#     # print('==============================')
#     #
#     # # report performance using RMSE
#     # rmse_valid = sqrt(mean_squared_error(raw_values[:, 0][len(train_scaled2) + 1:len(valid) + len(train_scaled2) + 1], predictions_valid))
#     # print('valid RMSE: %.5f' % rmse_valid)
#     # print('==============================')
#
#
#     return y_train_pred, y_val_pred, y_test_pred#, duration, training_itrs
#
#
#
#

