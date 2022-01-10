# GAF CNN Classifier
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

# def calculatePerformance(model, x):
#         from sklearn.metrics import mean_squared_error,mean_absolute_error
#         from math import sqrt
#         predictions = model.predict(x)
#         actuals = x.copy()
#         actuals = actuals.reshape(actuals.shape[0], actuals.shape[1])
#         predictions = predictions[:,1]
#         return mean_absolute_error(actuals, predictions), sqrt(mean_squared_error(actuals, predictions))
#======================================================================================================

def GAF_CNN_classifier(save_path, filename, x_train, y_train, x_val, y_val, x_test, y_test, nb_classes, run, verbose=False, min_exp_val_loss=0.005):
        np.random.seed()
        batch_size = 50
        nb_epochs = 500
        # CNN Model
        input_layer = Input(shape = x_train.shape[1:])
        x = layers.Conv2D(32, 3, 1, padding='same')(input_layer)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, 3, 1, padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2, padding='same')(x)
        x = layers.Conv2D(32, 3, 1, padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, 3, 1, padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2, padding='same')(x)
        x = layers.Conv2D(64, 3, 1, padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        output_layer = layers.Dense(nb_classes, activation='softmax')(x)
        model = models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),metrics=['acc'])
        if(verbose==True):
                model.summary()
        early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=min_exp_val_loss, patience=15)
        baseline_stop = TerminateOnBaseline(monitor='val_acc', baseline=1.0, patience=15)
        # reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.92, patience=3, min_lr=0.0001)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.92, patience=5, min_lr=0.0001)
        file_path = save_path + 'models/' + filename + '_GAF_CNN_best_model_run_' + str(run) + '.hdf5'
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
        hist_df.to_csv(save_path + 'models/' + filename + '_GAF_CNN_history_run_' + str(run) + '.csv', index=False)
        keras.backend.clear_session()

        if verbose ==1:
                val_perf = mean_absolute_error(y_val, y_val_pred), sqrt(mean_squared_error(y_val, y_val_pred))
                test_perf = mean_absolute_error(y_test, y_test_pred), sqrt(mean_squared_error(y_test, y_test_pred))
                print(f'MAE= {abs(val_perf[0] - test_perf[0]) < 0.01}, val: {val_perf[0]}, test: {test_perf[0]}')
                print(f'RMSE= {abs(val_perf[1] - test_perf[1]) < 0.01}, val: {val_perf[1]}, test: {test_perf[1]}')

        return y_train_pred, y_val_pred, y_test_pred, duration, training_itrs # y_train, y_val, hist
