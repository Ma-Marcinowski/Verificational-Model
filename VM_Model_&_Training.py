import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import losses
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

BatchSize = 64

TrainSeq = DataSequence(dataframe='/path/TrainDataframe.csv', batch_size = BatchSize)
ValidSeq = DataSequence(dataframe='/path/ValidDataframe.csv', batch_size = BatchSize)

left_input = Input(shape=(256, 256, 1), name='left_input')
right_input = Input(shape=(256, 256, 1), name='right_input')

xl = Conv2D(96, (11, 11), strides=4, padding='same', activation='relu', input_shape=(256, 256, 1), name='1stConvLeft')(left_input)
xl = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='1stPoolLeft')(xl)
xl = Conv2D(256, (5, 5), strides=1, padding='same', activation='relu', name='2ndConvLeft')(xl)
xl = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='2ndPoolLeft')(xl)
xl = Conv2D(384, (3, 3), strides=1, padding='same', activation='relu', name='3rdConvLeft')(xl)
xl = Conv2D(384, (3, 3), strides=1, padding='same', activation='relu', name='4thConvLeft')(xl)
xl = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu', name='5thConvLeft')(xl)
xl = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='3rdPoolLeft')(xl)
left_out = Flatten()(xl)

xr = Conv2D(96, (11, 11), strides=4, padding='same', activation='relu', input_shape=(256, 256, 1), name='1stConvRight')(right_input)
xr = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='1stPoolRight')(xr)
xr = Conv2D(256, (5, 5), strides=1, padding='same', activation='relu', name='2ndConvRight')(xr)
xr = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='2ndPoolRight')(xr)
xr = Conv2D(384, (3, 3), strides=1, padding='same', activation='relu', name='3rdConvRight')(xr)
xr = Conv2D(384, (3, 3), strides=1, padding='same', activation='relu', name='4thConvRight')(xr)
xr = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu', name='5thConvRight')(xr)
xr = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='3rdPoolRight')(xr)
right_out = Flatten()(xr)

x = keras.layers.concatenate([left_out, right_out], axis=1)
x = Dense(4096, activation='relu', name='1stFCL')(x) 
x = Dropout(rate=0.5)(x)
x = Dense(1024, activation='relu', name='2ndFCL')(x) 
x = Dropout(rate=0.5)(x)
x = Dense(256, activation='relu', name='3rdFCL')(x)
x = Dropout(rate=0.5)(x)
output = Dense(1, activation='sigmoid', name='output')(x) 

model = Model(inputs=[left_input, right_input], outputs=[output])

#model.load_weights('/path/checkpoint/VM_SNN_W.hdf5')
#model = load_model('/path/VM_SNN_M.h5')

sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

tensorboard = keras.callbacks.TensorBoard(log_dir='/path/logs/',
                                          histogram_freq=0,
                                          batch_size=BatchSize,
                                          write_graph=True,
                                          write_grads=False,
                                          write_images=False,
                                          update_freq='epoch')

checkpoint = keras.callbacks.ModelCheckpoint(filepath='/path/checkpoint/VM_SNN_W.hdf5', 
                                             monitor='val_loss', 
                                             verbose=1, 
                                             save_best_only=True, 
                                             save_weights_only=True, 
                                             mode='auto',
                                             period=1)

history = model.fit_generator(generator=TrainSeq,
                              validation_data=ValidSeq,
                              callbacks=[tensorboard, checkpoint],
                              use_multiprocessing=False, 
                              shuffle=False,
                              verbose=1,
                              validation_freq=1,
                              initial_epoch=0,
                              epochs=9)

#model.save('/path/VM_SNN_M.h5', overwrite=True, include_optimizer=True)

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
