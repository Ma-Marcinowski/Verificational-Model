import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.models import Model, load_model

def ConvLayerFilters(model_load_path, img_out_path, layer_name):

    model = load_model(model_load_path)

    conv_layer = model.get_layer(name=layer_name)
    conv_layer_weights = conv_layer.get_weights()

    num_of_filters = conv_layer_weights[0].shape[-1]
    num_of_channels = conv_layer_weights[0].shape[-2]

    filters_nulls = len(str(num_of_filters))
    channel_nulls = len(str(num_of_channels))

    print(layer_name, ' filters  to  visualise: ', num_of_channels * num_of_filters)

    filter_index = 0

    for filter_ind in tqdm(range(num_of_filters), desc='Filters visualisation:', leave=True):

        filter_index += 1

        channel_index = 0

        for channel_idx in tqdm(range(num_of_channels), desc='inner-loop:', leave=False):

            filter_matrix = conv_layer_weights[0][:, :, channel_idx, filter_ind]

            normalized = cv2.normalize(filter_matrix, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

            resized = cv2.resize(normalized,(256, 256))

            blur = cv2.GaussianBlur(resized,(5, 5),0)

            retv, ots = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            channel_index += 1

            ind = str(filter_index).zfill(filters_nulls)
            idx = str(channel_index).zfill(channel_nulls)

            cv2.imwrite(img_out_path + layer_name + '_filters_' + ind + '_' + idx + '.png', ots)

    print(layer_name, ' filters  visualised: ', filter_index * channel_index)

named_conv_layer_filter_weights_visualised = ConvLayerFilters(model_load_path='/saved/model/directory/model.h5',
                                                              img_out_path='/visualised/filters/directory/',
                                                              layer_name='x_conv_layer_to_visualise')
