import pandas as pd
import keras
from keras import optimizers, losses
from keras.models import Model, load_model
from keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, Dropout, Flatten

train_data = pd.read_csv('/path/TrainDataframe.csv')
valid_data = pd.read_csv('/path/ValidDataframe.csv')

trainsize = train_data.shape[0]
validsize = valid_data.shape[0]

batchsize = 32

def generator(df):

  datagen=keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

  left_generator=datagen.flow_from_dataframe(dataframe=df,
                                             x_col='Leftname',
                                             y_col='Label',
                                             target_size=(256, 256),
                                             color_mode='grayscale',
                                             class_mode='binary',
                                             batch_size=batchsize,
                                             shuffle=False,
                                             drop_duplicates=False)
  right_generator=datagen.flow_from_dataframe(dataframe=df,
                                              x_col='Rightname',
                                              y_col='Label',
                                              target_size=(256, 256),
                                              color_mode='grayscale',
                                              class_mode='binary',
                                              batch_size=batchsize,
                                              shuffle=False,
                                              drop_duplicates=False)
  while True:
      x1 = left_generator.next()
      x2 = right_generator.next()
      y0 = left_generator.next()
      yield ({'left_input': x1[0], 'right_input': x2[0]}, {'output': y0[1]})

left_input = Input(shape=(256, 256, 1), name='left_input')
right_input = Input(shape=(256, 256, 1), name='right_input')

xl = Conv2D(96, (11, 11), strides=4, padding='same', activation='relu', input_shape=(256, 256, 1), name='FirstConvLeft')(left_input)
xl = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='FirstPoolLeft')(xl)
xl = Conv2D(256, (5, 5), strides=1, padding='same', activation='relu', name='SecondConvLeft')(xl)
xl = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='SecondPoolLeft')(xl)
xl = Conv2D(384, (3, 3), strides=1, padding='same', activation='relu', name='ThirdConvLeft')(xl)
xl = Conv2D(384, (3, 3), strides=1, padding='same', activation='relu', name='FourthConvLeft')(xl)
xl = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu', name='FifthConvLeft')(xl)
xl = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='ThirdPoolLeft')(xl)
left_out = Flatten()(xl)

xr = Conv2D(96, (11, 11), strides=4, padding='same', activation='relu', input_shape=(256, 256, 1), name='FirstConvRight')(right_input)
xr = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='FirstPoolRight')(xr)
xr = Conv2D(256, (5, 5), strides=1, padding='same', activation='relu', name='SecondConvRight')(xr)
xr = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='SecondPoolRight')(xr)
xr = Conv2D(384, (3, 3), strides=1, padding='same', activation='relu', name='ThirdConvRight')(xr)
xr = Conv2D(384, (3, 3), strides=1, padding='same', activation='relu', name='FourthConvRight')(xr)
xr = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu', name='FifthConvRight')(xr)
xr = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='ThirdPoolRight')(xr)
right_out = Flatten()(xr)

x = keras.layers.concatenate([left_out, right_out], axis=1)
x = Dense(8192, activation='relu', name='FirstFCL')(x)
x = Dropout(0.4)(x)
x = Dense(8192, activation='relu', name='SecondFCL')(x)
x = Dropout(0.4)(x)
output = Dense(1, activation='sigmoid', name='output')(x)

model = Model(inputs=[left_input, right_input], outputs=[output])

model.compile(optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy', 'binary_accuracy'])

tensorboard = keras.callbacks.TensorBoard(log_dir='/path/logs',
                                          histogram_freq=0,
                                          batch_size=batchsize,
                                          write_graph=True,
                                          write_grads=True,
                                          write_images=True,
                                          #embeddings_freq=1,
                                          #embeddings_layer_names=['FirstFCL', 'SecondFCL', 'output'],
                                          #embeddings_metadata=None,
                                          #embeddings_data=[train_generator],
                                          update_freq='epoch')

model.fit_generator(generator=generator(train_data),
                    steps_per_epoch=trainsize//batchsize,
                    validation_data=generator(valid_data),
                    validation_steps=validsize//batchsize,
                    callbacks=[tensorboard],
                    use_multiprocessing=False,
                    shuffle=False,
                    validation_freq=1,
                    verbose=1,
                    epochs=9,
                    initial_epoch=0)

model.save('/path/VM_SNN.h5', overwrite=True, include_optimizer=True)
