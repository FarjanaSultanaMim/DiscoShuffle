
import os
import pickle

from keras import callbacks

class NBatchLogger(callbacks.Callback):
    def __init__(self, out_file):
        self.logs = []
        self.logs_nb = []
        self.out_file = out_file

    def update(self):
        with open(self.out_file, "wb") as f:
            pickle.dump({"logs_b": self.logs, "logs_e": self.logs_nb}, f)

    def on_batch_end(self, batch, logs={}):
        self.logs += [logs]
        self.update()

    def on_epoch_end(self, batch, logs={}):
        self.logs_nb += [logs]
        self.update()

        