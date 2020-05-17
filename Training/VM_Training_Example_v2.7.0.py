import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense
from tensorflow.keras.layers import BatchNormalization, GaussianDropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_image(img):
    return img_to_array(load_img(img, color_mode='grayscale')) / 255.

class TrainDataSequence(tf.keras.utils.Sequence):

    def __init__(self, dataframe, batch_size):
        self.df = pd.read_csv(dataframe)
        self.batch_size = batch_size

        self.labels = self.df['Label'].tolist()
        self.weights = self.df['Weight'].tolist()
        self.leftnames = self.df['Leftname'].tolist()
        self.rightnames = self.df['Rightname'].tolist()

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def get_batch_labels(self, idx):
        batch_labels = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        return np.array(batch_labels)
    
    def get_batch_weights(self, idx):
        batch_weights = self.weights[idx * self.batch_size: (idx + 1) * self.batch_size]
        return np.array(batch_weights)

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
        batch_w = self.get_batch_weights(idx)
        return ({'left_input': batch_x1, 'right_input': batch_x2}, {'output': batch_y}, batch_w)
    
class ValidDataSequence(tf.keras.utils.Sequence):

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

TrainSeq = TrainDataSequence(dataframe='/path/TrainDataframe.csv', batch_size = BatchSize)
ValidSeq = ValidDataSequence(dataframe='/path/ValidDataframe.csv', batch_size = BatchSize)

left_input = Input(shape=(256, 256, 1), name='left_input')
right_input = Input(shape=(256, 256, 1), name='right_input')

def CosineDistance(vectors):

    cos_dist = tf.losses.cosine_distance(vectors[0], vectors[1], axis=1, reduction=tf.losses.Reduction.NONE)

    return cos_dist

def EuclideanDistance(vectors):

    euc_dist = tf.math.sqrt(tf.math.reduce_sum(tf.math.squared_difference(vectors[0], vectors[1]), axis=1, keepdims=True))

    return euc_dist

def CoreNet(ix, path):

    x = Conv2D(16, (12, 12), strides=1, padding='same', activation='relu', name='1stConv' + path)(ix)
    x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='1stPool' + path)(x)

    x = Conv2D(32, (6, 6), strides=1, padding='same', activation='relu', name='2ndConv' + path)(x)
    x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='2ndPool' + path)(x)

    x = Conv2D(64, (3, 3), strides=1, padding='same', activation='relu', name='3rdConv' + path)(x)
    x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)
    ox = tf.keras.layers.GlobalAveragePooling2D(name=path + 'GAP')(x)

    return ox

left_out = CoreNet(ix=left_input, path='Left')
right_out = CoreNet(ix=right_input, path='Right')

c = tf.keras.layers.Lambda(CosineDistance, name="CosineDistance")([left_out, right_out])
e = tf.keras.layers.Lambda(EuclideanDistance, name="EuclideanDistance")([left_out, right_out])

x = tf.keras.layers.concatenate([left_out, right_out], axis=-1)

x = GaussianDropout(rate=0.5)(x)
x = Dense(1024, activation='relu', name='1stFCL')(x)
x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)

x = GaussianDropout(rate=0.5)(x)
x = Dense(512, activation='relu', name='2ndFCL')(x)
x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)

x = GaussianDropout(rate=0.5)(x)
x = Dense(256, activation='relu', name='3rdFCL')(x)
x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)

x = GaussianDropout(rate=0.5)(x)
x = Dense(1, activation='relu', name='4thFCL')(x)
f = BatchNormalization(axis=-1, scale=True, trainable=True)(x)

i = tf.keras.layers.concatenate([f, c, e], axis=-1)

output = Dense(1, activation='sigmoid', name='output')(i)

model = Model(inputs=[left_input, right_input], outputs=[output])

#model = load_model('/path/VM/VM.h5')

Adam = tf.keras.optimizers.Adam(learning_rate=0.001,
                                beta_1=0.9,
                                beta_2=0.999,
                                epsilon=1e-08)

model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=['accuracy'])

csv_logger = tf.keras.callbacks.CSVLogger('/path/TrainLog.csv', separator=',', append=False)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='/path/TrainLogs/',
                                             histogram_freq=0,
                                             batch_size=BatchSize,
                                             write_graph=True,
                                             update_freq='batch')

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='/path/VM-{epoch:02d}-{val_loss:.2f}.h5',
                                                load_weights_on_restart=False,
                                                monitor='val_loss',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=False,
                                                mode='min',
                                                save_freq='epoch')

reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.1,
                                               patience=1,
                                               verbose=1,
                                               mode='min',
                                               min_delta=0.0001,
                                               cooldown=1,
                                               min_lr=1e-8)

history = model.fit(x=TrainSeq,
                    validation_data=ValidSeq,
                    callbacks=[csv_logger, checkpoint, reduceLR, tensorboard], 
                    use_multiprocessing=False,
                    shuffle=True,
                    max_queue_size=10,
                    workers=1,
                    verbose=1,
                    validation_freq=1,
                    initial_epoch=0,
                    epochs=1)

#model.save('/path/VM.h5', overwrite=True, include_optimizer=True)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
