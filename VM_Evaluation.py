import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import optimizers, losses
from keras.utils import Sequence
from keras.models import Model, load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, Dropout, Flatten

def load_image(img):
    return img_to_array(load_img(img, color_mode='grayscale')) / 255.

class DataSequence(Sequence):

    def __init__(self, dataframe, batch_size):
        self.df = pd.read_csv(dataframe)
        self.batch_size = batch_size

        self.labels = self.df['Label'].tolist()
        self.leftnames = self.df['Leftname'].tolist()
        self.rightnames = self.df['Rightname'].tolist()

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def get_batch_labels(self, idx):
        batch_labels = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        return np.array(batch_labels)

    def get_batch_leftnames(self, idx):
        batch_leftnames = self.leftnames[idx * self.batch_size: (idx + 1) * self.batch_size]
        return np.array([load_image(i) for i in batch_leftnames])

    def get_batch_rightnames(self, idx):
        batch_rightnames = self.rightnames[idx * self.batch_size: (idx + 1) * self.batch_size]
        return np.array([load_image(j) for j in batch_rightnames])

    def __getitem__(self, idx):
        batch_x1 = self.get_batch_leftnames(idx)
        batch_x2 = self.get_batch_rightnames(idx)
        batch_y = self.get_batch_labels(idx)
        return ({'left_input': batch_x1, 'right_input': batch_x2}, {'output': batch_y})

BatchSize = 0

TestSeq = DataSequence(dataframe='/path/TestDataframe.csv', batch_size = BatchSize)

#model.load_weights('/path/VM_SNN_W.h5')
#model = load_model('/path/VM_SNN_M.h5')

tensorboard = keras.callbacks.TensorBoard(log_dir='/content/logs',
                                          histogram_freq=1,
                                          batch_size=BatchSize,
                                          write_graph=True,
                                          write_grads=True,
                                          write_images=True,
                                          update_freq='epoch')

evaluation = model.evaluate_generator(generator=TestSeq,
                                      callbacks=[tensorboard],
                                      max_queue_size=10,
                                      workers=1,
                                      use_multiprocessing=False,
                                      verbose=1)
print(evaluation)
print(model.metrics_names)
