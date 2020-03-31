import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense
from tensorflow.keras.layers import BatchNormalization, GaussianDropout

left_input = Input(shape=(256, 256, 1), name='left_input')
right_input = Input(shape=(256, 256, 1), name='right_input')

def CosineDistance(vectors):
  
    cos_dist = tf.compat.v1.losses.cosine_distance(vectors[0], vectors[1], axis=1, reduction=tf.losses.Reduction.NONE)

    return cos_dist

def EuclideanDistance(vectors):
  
    euc_dist = tf.math.sqrt(tf.math.reduce_sum(tf.math.squared_difference(vectors[0], vectors[1]), axis=1, keepdims=True))
    
    return euc_dist

def CoreNet(ix, path):

    x = Conv2D(32, (16, 16), strides=1, dilation_rate=2, padding='same', activation='relu', name=path + 'Conv')(ix)
    x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)
    ox = tf.keras.layers.GlobalAveragePooling2D(name=path + 'GAP')(x)

    return ox

left_out = CoreNet(ix=left_input, path='Left')
right_out = CoreNet(ix=right_input, path='Right')

c = tf.keras.layers.Lambda(CosineDistance, name="CosineDistance")([left_out, right_out])
e = tf.keras.layers.Lambda(EuclideanDistance, name="EuclideanDistance")([left_out, right_out])

x = tf.keras.layers.concatenate([left_out, right_out], axis=-1)

x = GaussianDropout(rate=0.5)(x)                     
x = Dense(512, activation='relu', name='1stFCL')(x) 
x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)

x = GaussianDropout(rate=0.5)(x)                    
x = Dense(256, activation='relu', name='2ndFCL')(x) 
x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)

x = GaussianDropout(rate=0.5)(x)                    
x = Dense(128, activation='relu', name='3rdFCL')(x) 
x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)

x = GaussianDropout(rate=0.5)(x)                    
x = Dense(1, activation='relu', name='4thFCL')(x) 
f = BatchNormalization(axis=-1, scale=True, trainable=True)(x)

i = tf.keras.layers.concatenate([f, c, e], axis=-1)

output = Dense(1, activation='sigmoid', name='output')(i)

model = Model(inputs=[left_input, right_input], outputs=[output])

model.summary(line_length=None,
              positions=None,
              print_fn=None)
tf.keras.utils.plot_model(model, to_file='/plot/save/path/VM_v3.6.1.png', show_shapes=True, show_layer_names=True, rankdir='TB')
