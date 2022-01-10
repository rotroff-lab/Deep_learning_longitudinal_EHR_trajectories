# ResNet Classifier
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
from sklearn.metrics import mean_squared_error,mean_absolute_error
from math import sqrt
#======================================================================================================

def ResNet_classifier(save_path, filename, x_train, y_train, x_val, y_val, x_test, y_test, nb_classes, run, verbose=False, min_exp_val_loss=0.005):
        np.random.seed()
        batch_size = 50
        nb_epochs = 500
        n_feature_maps = 64
        # ResNet Model
        # Block 1
        input_layer = Input(shape = x_train.shape[1:])
        x = layers.Conv1D(n_feature_maps, 8, 1, padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(n_feature_maps, 5, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(n_feature_maps, 3, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        residual = layers.Conv1D(n_feature_maps, 1, 1, padding='same')(input_layer)
        residual = layers.BatchNormalization()(residual)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        previous_block_activation = x
        # Block 2
        x = layers.Conv1D(n_feature_maps, 8, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(n_feature_maps, 5, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(n_feature_maps, 3, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        residual = layers.BatchNormalization()(previous_block_activation)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        previous_block_activation = x
        # Block 3
        x = layers.Conv1D(n_feature_maps, 8, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(n_feature_maps, 5, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(n_feature_maps, 3, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        residual = layers.BatchNormalization()(previous_block_activation)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling1D()(x)
        output_layer = layers.Dense(nb_classes, activation='softmax')(x)
        model = models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),metrics=['acc'])
        if(verbose==True):
                model.summary()
        early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=min_exp_val_loss, patience=15)
        baseline_stop = TerminateOnBaseline(monitor='val_acc', baseline=1.0, patience=15)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.92, patience=3, min_lr=0.0001)
        file_path = save_path + 'models/' + filename + '_ResNet_best_model_run_' + str(run) + '.hdf5'
        model_checkpoint = callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
        mini_batch_size = int(min(x_train.shape[0]/10, batch_size))
        start_time = time.time()
        hist = model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                         validation_data=(x_val, y_val), verbose=2,
                         callbacks=[early_stop, baseline_stop, reduce_lr, model_checkpoint])
        duration = time.time() - start_time
        print("Elapsed Training Time: %f" % (duration))
        model = models.load_model(file_path)
        y_val_pred = model.predict(x_val)
        y_test_pred = model.predict(x_test)
        y_train_pred = model.predict(x_train)
        training_itrs = len(hist.history['loss'])
        hist_df = pd.DataFrame(hist.history)
        hist_df.to_csv(save_path + 'models/' + filename + '_ResNet_history_run_' + str(run) + '.csv', index=False)
        keras.backend.clear_session()

        if verbose == 1:
                val_perf = mean_absolute_error(y_val, y_val_pred), sqrt(mean_squared_error(y_val, y_val_pred))
                test_perf = mean_absolute_error(y_test, y_test_pred), sqrt(mean_squared_error(y_test, y_test_pred))
                print(f'MAE= {abs(val_perf[0] - test_perf[0]) < 0.01}, val: {val_perf[0]}, test: {test_perf[0]}')
                print(f'RMSE= {abs(val_perf[1] - test_perf[1]) < 0.01}, val: {val_perf[1]}, test: {test_perf[1]}')

        return y_train_pred, y_val_pred, y_test_pred, duration, training_itrs
