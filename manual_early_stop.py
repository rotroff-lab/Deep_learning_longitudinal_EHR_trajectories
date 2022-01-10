import tensorflow 
from tensorflow import keras
from tensorflow.keras import callbacks

class TerminateOnBaseline(tensorflow.keras.callbacks.Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline"""
    def __init__(self, monitor='acc', baseline=0.9, patience=0):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.wait = 0
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc < self.baseline:
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    #print('Epoch %d: Reached baseline, terminating training' % (epoch))
                    self.model.stop_training = True
