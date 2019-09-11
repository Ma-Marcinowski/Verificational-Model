import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense
from tensorflow.keras.layers import BatchNormalization, GaussianDropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_image(img):
    return img_to_array(load_img(img, color_mode='grayscale')) / 255.

class DataSequence(tf.keras.utils.Sequence):

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

BatchSize = 16

TestSeq = DataSequence(dataframe='/path/TestDataframe.csv', batch_size = BatchSize)

#model = load_model('/path/VM-{epoch:02d}-{val_loss:.2f}.h5')
#model = load_model('/path/VM.h5')

Adam = tf.keras.optimizers.Adam(learning_rate=0.001,
                                beta_1=0.9, 
                                beta_2=0.999, 
                                epsilon=1e-08)

TP  = tf.keras.metrics.TruePositives()
TN  = tf.keras.metrics.TrueNegatives()
FP  = tf.keras.metrics.FalsePositives()
FN  = tf.keras.metrics.FalseNegatives()
Rec = tf.keras.metrics.Recall()
Pre = tf.keras.metrics.Precision()
AUC = tf.keras.metrics.AUC()

model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=['accuracy', TP, TN, FP, FN, Rec, Pre, AUC])

csv_logger = tf.keras.callbacks.CSVLogger('/path/TestLog.csv', separator=',', append=False)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='/path/TestLogs/',
                                             histogram_freq=1,
                                             batch_size=BatchSize,
                                             write_graph=True,
                                             write_grads=True,
                                             write_images=True,
                                             update_freq='batch')

evaluation = model.evaluate_generator(generator=TestSeq,
                                      callbacks=[csv_logger, tensorboard],
                                      use_multiprocessing=False,
                                      max_queue_size=10,
                                      workers=1,
                                      verbose=1)
print(evaluation)
print(model.metrics_names)
