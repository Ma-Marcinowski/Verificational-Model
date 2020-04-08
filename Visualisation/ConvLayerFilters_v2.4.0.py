import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.models import Model, load_model

def ConvLayerFilters(model_load_path, img_out_path, layer_name):

    model = load_model(model_load_path)

    conv_layer = model.get_layer(name=layer_name)
    conv_layer_weights = conv_layer.get_weights()

    num_of_filters = conv_layer_weights[0].shape[-1]

    nulls = len(str(num_of_filters))

    print(layer_name, ' filters  to  visualise: ', num_of_filters)

    filter_index = 0

    while filter_index < num_of_filters:

        filter_matrix = conv_layer_weights[0][:, :, 0, filter_index]

        normalized = cv2.normalize(filter_matrix, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

        resized = cv2.resize(normalized,(256,256))

        blur = cv2.GaussianBlur(resized,(5,5),0)

        retv, ots = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        filter_index += 1
        
        ind = str(filter_index).zfill(nulls)

        cv2.imwrite(img_out_path + layer_name + '_filter_' + ind + '.png', ots)

        print('%.2f%%'%(100*filter_index/num_of_filters), end="\r")

    print(layer_name, ' filters  visualised: ', filter_index)

named_conv_layer_filter_weights_visualised = ConvLayerFilters(model_load_path='/saved/model/directory/model.h5',
                                                              img_out_path='/visualised/filters/directory/',
                                                              layer_name='x_conv_layer_to_visualise')
