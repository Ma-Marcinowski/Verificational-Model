import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, load_model

def ConvLayerWeights(model_load_path, img_out_path, layer_name):

    model = load_model(model_load_path)

    conv_layer = model.get_layer(name=layer_name)
    conv_layer_weights = conv_layer.get_weights()

    f = conv_layer_weights[0].shape[-1]

    nulls = len(str(f))

    print(layer_name, ' filter  weights  to  visualise: ', f)

    i = 0

    while i < f:

        filter_matrix = conv_layer_weights[0][:, :, 0, i]

        normalized = cv2.normalize(filter_matrix, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

        resized = cv2.resize(normalized,(256,256))

        blur = cv2.GaussianBlur(resized,(5,5),0)

        retv, ots = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        i += 1
        ind = str(i).zfill(nulls)

        cv2.imwrite(img_out_path + layer_name + '_filter_weights_' + ind + '.png', ots)

        print('%.2f%%'%(100*i/f), end="\r")

    print(layer_name, ' filter  weights  visualised: ', i)

named_conv_weights = ConvLayerWeights(model_load_path='/saved/model/directory/model.h5',
                                      img_out_path='/visualised/weights/directory/',
                                      layer_name='x_conv_layer_to_visualise')
