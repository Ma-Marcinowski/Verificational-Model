import keras
from keras.models import Model
from keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, Dropout, Flatten

left_input = Input(shape=(256, 256, 1), name='left_input')
right_input = Input(shape=(256, 256, 1), name='right_input')

xl = Conv2D(64, (3, 3), strides=1, padding='same', activation='relu', input_shape=(256, 256, 1), name='1stConvLeft')(left_input)
xl = Conv2D(64, (3, 3), strides=1, padding='same', activation='relu', name='2ndConvLeft')(xl)
xl = MaxPooling2D(pool_size=(3, 3), strides=3, padding='valid', name='1stPoolLeft')(xl)
xl = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', name='3rdConvLeft')(xl)
xl = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', name='4thConvLeft')(xl)
xl = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name='2ndPoolLeft')(xl)
xl = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu', name='5thConvLeft')(xl)
xl = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu', name='6thConvLeft')(xl)
xl = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name='3rdPoolLeft')(xl)
xl = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu', name='7thConvLeft')(xl)
xl = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu', name='8thConvLeft')(xl)
xl = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name='4thPoolLeft')(xl)
xl = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu', name='9thConvLeft')(xl)
xl = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu', name='10thConvLeft')(xl)
xl = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name='5thPoolLeft')(xl)
left_out = Flatten()(xl)

xr = Conv2D(64, (3, 3), strides=1, padding='same', activation='relu', input_shape=(256, 256, 1), name='1stConvRight')(right_input)
xr = Conv2D(64, (3, 3), strides=1, padding='same', activation='relu', name='2ndConvRight')(xr)
xr = MaxPooling2D(pool_size=(3, 3), strides=3, padding='valid', name='1stPoolRight')(xr)
xr = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', name='3rdConvRight')(xr)
xr = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', name='4thConvRight')(xr)
xr = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name='2ndPoolRight')(xr)
xr = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu', name='5thConvRight')(xr)
xr = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu', name='6thConvRight')(xr)
xr = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name='3rdPoolRight')(xr)
xr = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu', name='7thConvRight')(xr)
xr = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu', name='8thConvRight')(xr)
xr = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name='4thPoolRight')(xr)
xr = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu', name='9thConvRight')(xr)
xr = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu', name='10thConvRight')(xr)
xr = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name='5thPoolRight')(xr)
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

keras.utils.print_summary(model,
                          line_length=None,
                          positions=None,
                          print_fn=None)
keras.utils.plot_model(model, to_file='/content/SNN_VGG16.png', show_shapes=True, show_layer_names=True, rankdir='TB')
