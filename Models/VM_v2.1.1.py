import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense
from tensorflow.keras.layers import BatchNormalization, GaussianDropout

left_input = Input(shape=(256, 256, 1), name='left_input')
right_input = Input(shape=(256, 256, 1), name='right_input')

def CoreNet(ix, path):

    x = Conv2D(256, (12, 12), strides=4, padding='same', activation='relu', name='1stConv' + path)(ix)
    x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='1stPool' + path)(x)

    x = Conv2D(512, (6, 6), strides=2, padding='same', activation='relu', name='2ndConv' + path)(x)
    x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='2ndPool' + path)(x)

    x = Conv2D(1024, (3, 3), strides=1, padding='same', activation='relu', name='3rdConv' + path)(x)
    x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)
    ox = tf.keras.layers.GlobalAveragePooling2D(name=path + 'GAP')(x)

    return ox

left_out = CoreNet(ix=left_input, path='Left')
right_out = CoreNet(ix=right_input, path='Right')

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
output = Dense(1, activation='sigmoid', name='output')(x)

model = Model(inputs=[left_input, right_input], outputs=[output])

model.summary(line_length=None,
              positions=None,
              print_fn=None)
tf.keras.utils.plot_model(model, to_file='/path/VM_v2.1.1.png', show_shapes=True, show_layer_names=True, rankdir='TB')
