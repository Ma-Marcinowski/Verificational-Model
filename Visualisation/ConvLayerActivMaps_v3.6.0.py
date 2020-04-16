import tensorflow as tf
import numpy as np
import cv2

from tqdm import tqdm_notebook as tqdm

from tensorflow.keras.models import Model, load_model

tf.enable_eager_execution()

def InitializeImage(img_in_path):

    tensor2 = cv2.imread(img_in_path, 0).astype('float32')
    tensor3 = tf.keras.backend.expand_dims(tensor2, axis=-1)
    tensor4 = tf.keras.backend.expand_dims(tensor3, axis=0) / 255.

    return tensor2, tensor4

def ConvLayerActivMaps(model_load_path, img_in_path, img_out_path, input_layer_name, conv_layer_name, num_of_filters):

    raw_image, img_tensor = InitializeImage(img_in_path=img_in_path)

    model = load_model(model_load_path)

    conv_layer = Model(inputs=model.get_layer(name=input_layer_name).input, outputs=model.get_layer(name=conv_layer_name).output)

    conv_layer_output = conv_layer(img_tensor)

    nulls = len(str(num_of_filters))

    empty_image = np.zeros(raw_image.shape, dtype=np.float32)

    for filter_index in tqdm(range(num_of_filters), desc=conv_layer_name + ' activ. maps visualisation:', leave=True):

        act_tensor = conv_layer_output[:, :, :, filter_index].numpy()

        act_matrix = act_tensor[0, :, :]

        normalized = cv2.normalize(act_matrix, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        resized = cv2.resize(normalized, raw_image.shape)

        bgr_act_map = cv2.merge((empty_image, resized, empty_image))
        bgr_raw_img = cv2.merge((raw_image, resized, empty_image))

        bordered_bgr = cv2.copyMakeBorder(bgr_act_map, top=4, bottom=4, left=4, right=2, borderType=cv2.BORDER_CONSTANT, value=255)
        bordered_raw = cv2.copyMakeBorder(bgr_raw_img, top=4, bottom=4, left=2, right=4, borderType=cv2.BORDER_CONSTANT, value=255)
        
        both = np.concatenate((bordered_bgr, bordered_raw), axis=1)

        ind = str(filter_index + 1).zfill(nulls)

        cv2.imwrite(img_out_path + conv_layer_name + '_activation_map_' + ind + '.png', both)

visexp_activ_maps = ConvLayerActivMaps(model_load_path='/saved/model/directory/model.h5',
                                       img_out_path='/activation/maps/save/directory/',
                                       img_in_path='input/image/path.png',
                                       input_layer_name='model_input_layer',
                                       conv_layer_name='x_conv_layer_to_output_activation_maps',
                                       num_of_features=1024) #Number of the named gap layer filters.
