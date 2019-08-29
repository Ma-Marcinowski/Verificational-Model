import pandas as pd
import keras
from keras import optimizers, losses
from keras.models import Model, load_model
from keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, Dropout, Flatten

test_data = pd.read_csv('/path/TestDataframe.csv')

testsize = test_data.shape[0]

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
      y0 = right_generator.next()
      yield ({'left_input': x1[0], 'right_input': x2[0]}, {'output': y0[1]})

model = load_model('/path/VM_SNN.h5')

tensorboard = keras.callbacks.TensorBoard(log_dir='/path/logs',
                                          histogram_freq=1,
                                          batch_size=batchsize,
                                          write_graph=True,
                                          write_grads=True,
                                          write_images=True,
                                          #embeddings_freq=1,
                                          #embeddings_layer_names=['FirstFCL', 'SecondFCL', 'output'],
                                          #embeddings_metadata=None,
                                          #embeddings_data=[left_test_generator, right_test_generator],
                                          update_freq='epoch')

model.evaluate_generator(generator=generator(test_data),
                        steps=testsize//batchsize
                        callbacks=[TensorBoard],
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False,
                        verbose=0)
