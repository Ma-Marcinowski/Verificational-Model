import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout 

left_input = Input(shape=(256, 256, 1), name='left_input')
right_input = Input(shape=(256, 256, 1), name='right_input')

xl = Conv2D(96, (11, 11), strides=4, padding='same', activation='relu', input_shape=(256, 256, 1), name='1stConvLeft')(left_input)
xl = BatchNormalization(axis=-1, scale=True, trainable=True)(xl)
xl = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='1stPoolLeft')(xl)
xl = Conv2D(256, (5, 5), strides=1, padding='same', activation='relu', name='2ndConvLeft')(xl)
xl = BatchNormalization(axis=-1, scale=True, trainable=True)(xl)
xl = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='2ndPoolLeft')(xl)
xl = Conv2D(384, (3, 3), strides=1, padding='same', activation='relu', name='3rdConvLeft')(xl)
xl = BatchNormalization(axis=-1, scale=True, trainable=True)(xl)
xl = Conv2D(384, (3, 3), strides=1, padding='same', activation='relu', name='4thConvLeft')(xl)
xl = BatchNormalization(axis=-1, scale=True, trainable=True)(xl)
xl = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu', name='5thConvLeft')(xl)
xl = BatchNormalization(axis=-1, scale=True, trainable=True)(xl)
xl = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='3rdPoolLeft')(xl)
left_out = Flatten()(xl)

xr = Conv2D(96, (11, 11), strides=4, padding='same', activation='relu', input_shape=(256, 256, 1), name='1stConvRight')(right_input)
xr = BatchNormalization(axis=-1, scale=True, trainable=True)(xr)
xr = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='1stPoolRight')(xr)
xr = Conv2D(256, (5, 5), strides=1, padding='same', activation='relu', name='2ndConvRight')(xr)
xr = BatchNormalization(axis=-1, scale=True, trainable=True)(xr)
xr = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='2ndPoolRight')(xr)
xr = Conv2D(384, (3, 3), strides=1, padding='same', activation='relu', name='3rdConvRight')(xr)
xr = BatchNormalization(axis=-1, scale=True, trainable=True)(xr)
xr = Conv2D(384, (3, 3), strides=1, padding='same', activation='relu', name='4thConvRight')(xr)
xr = BatchNormalization(axis=-1, scale=True, trainable=True)(xr)
xr = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu', name='5thConvRight')(xr)
xr = BatchNormalization(axis=-1, scale=True, trainable=True)(xr)
xr = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='3rdPoolRight')(xr)
right_out = Flatten()(xr)

x = tf.keras.layers.concatenate([left_out, right_out], axis=1)
x = Dense(4096, activation='relu', name='1stFCL')(x)                           
x = Dropout(rate=0.2)(x)                                                       
x = Dense(1024, activation='relu', name='2ndFCL')(x)
x = Dropout(rate=0.2)(x) 
x = Dense(256, activation='relu', name='3rdFCL')(x)
x = Dropout(rate=0.2)(x) 
output = Dense(1, activation='sigmoid', name='output')(x)                       

model = Model(inputs=[left_input, right_input], outputs=[output])


model.summary(line_length=None, 
              positions=None,
              print_fn=None)
tf.keras.utils.plot_model(model, to_file='/path/SNN-AlexNet.png', show_shapes=True, show_layer_names=True, rankdir='TB')
