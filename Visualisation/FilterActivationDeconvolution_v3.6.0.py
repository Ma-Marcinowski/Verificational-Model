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

def Deconvolution(filter_tensor, input_tensor):

    activation_map = tf.nn.relu(tf.keras.backend.conv2d(x=input_tensor,
                                                        kernel=filter_tensor,
                                                        strides=4,
                                                        padding='same'))

    activation_map_deconvolved = tf.nn.relu(tf.keras.backend.conv2d_transpose(x=activation_map,
                                                                              kernel=filter_tensor,
                                                                              output_shape=(1, 256, 256, 1),
                                                                              strides=(4, 4),
                                                                              padding='same'))

    return activation_map_deconvolved

def FilterActivationDeconvolution(model_load_path, img_in_path, img_out_path, layer_name):

    conv_model = load_model(model_load_path)
    conv_layer = conv_model.get_layer(name=layer_name)
    conv_layer_weights = conv_layer.get_weights()

    num_of_filters = conv_layer_weights[0].shape[-1]
    nulls = len(str(num_of_filters))

    raw_image, input_tensor = InitializeImage(img_in_path=img_in_path)
    empty_image = np.zeros(raw_image.shape, dtype=np.float32)

    for filter_index in tqdm(range(num_of_filters), desc='Features deconvolution:', leave=True):

        filter_matrix = conv_layer_weights[0][:, :, 0, filter_index]
        filter_tensor = tf.keras.backend.expand_dims(tf.keras.backend.expand_dims(filter_matrix, axis=-1), axis=-1)

        deconvolved = Deconvolution(filter_tensor=filter_tensor,
                                    input_tensor=input_tensor)


        deconvolved_tensor = deconvolved.numpy()
        deconvolved_matrix = deconvolved_tensor[0, :, :, 0]

        normalized = cv2.normalize(deconvolved_matrix, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        thv, filter_img = cv2.threshold(raw_image, 55, 255, cv2.THRESH_TOZERO)
        filtered = np.multiply(normalized, filter_img)

        bgr_deconv_img = cv2.merge((raw_image, filtered, empty_image))
        nnn_deconv_img = cv2.merge((empty_image, normalized, empty_image))

        bordered_bgr = cv2.copyMakeBorder(bgr_deconv_img, top=4, bottom=4, left=2, right=4, borderType=cv2.BORDER_CONSTANT, value=255)
        bordered_nnn = cv2.copyMakeBorder(nnn_deconv_img, top=4, bottom=4, left=4, right=2, borderType=cv2.BORDER_CONSTANT, value=255)
        both = np.concatenate((bordered_nnn, bordered_bgr), axis=1)

        ind = str(filter_index + 1).zfill(nulls)

        cv2.imwrite(img_out_path + 'deconvolved_' + ind + '.png', both)

filters_deconv = FilterActivationDeconvolution(model_load_path='/saved/model/directory/model.h5',
                                               img_in_path='input/image/path.png',
                                               img_out_path='/deconvolved/filter/activations/directory/',
                                               layer_name='x_conv_layer_to_deconvolve')
