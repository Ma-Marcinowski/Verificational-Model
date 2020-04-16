import tensorflow as tf
import numpy as np
import cv2
import os

from tqdm import tqdm_notebook as tqdm

from tensorflow.keras.models import Model, load_model

tf.enable_eager_execution()

def InitializeImage(img_path):

    tensor2 = cv2.imread(img_path, 0).astype('float32')
    tensor3 = tf.keras.backend.expand_dims(tensor2, axis=-1)
    tensor4 = tf.keras.backend.expand_dims(tensor3, axis=0) / 255.

    return tensor2, tensor4

def FeatureGradients(image, feature_index, feature_layer):

    with tf.GradientTape() as tape:
      
        tape.watch(image)

        feature_layer_output = feature_layer(image)

        given_feature_value = feature_layer_output[:, feature_index]
        
    feature_gradients = tape.gradient(given_feature_value, image)

    positive_gradients = tf.clip_by_value(feature_gradients, clip_value_min=0, clip_value_max=255)
    negative_gradients = tf.clip_by_value(feature_gradients, clip_value_min=-255, clip_value_max=0)

    small_postivie_gradients = tf.math.divide(1, tf.math.add(positive_gradients, 1e-9))
    small_negative_gradients = tf.math.divide(1, tf.math.add(negative_gradients, 1e-9))

    return small_postivie_gradients, small_negative_gradients, given_feature_value

def BackpropSaliencyMapping(model_load_path, img_out_path, images_dir, input_layer_name, output_layer_name, num_of_features):

    model = load_model(model_load_path)

    feature_layer = Model(inputs=model.get_layer(name=input_layer_name).input, outputs=model.get_layer(name=output_layer_name).output)

    nulls = len(str(num_of_features))

    image_names = os.listdir(images_dir)
    image_paths = [images_dir + n for n in image_names]
    images = zip(image_paths, image_names)

    for img_path, img_name in tqdm(images, total=len(image_paths), desc=output_layer_name + ' features mapping:', leave=True):

        raw_image, initial_image = InitializeImage(img_path=img_path)
        #empty_img = np.zeros(raw_image.shape, dtype=np.float32)

        for feature_index in tqdm(range(num_of_features), desc='Image features mapping:', leave=False):

            positive_feature_gradients, negative_feature_gradients, feature_value = FeatureGradients(image=initial_image, 
                                                                                                    feature_index=feature_index,
                                                                                                    feature_layer=feature_layer)

            pos_grads_tensor = positive_feature_gradients.numpy()
            pos_grads_matrix = pos_grads_tensor[0, :, :, 0]

            neg_grads_tensor = negative_feature_gradients.numpy()
            neg_grads_matrix = neg_grads_tensor[0, :, :, 0]

            value_scalar = feature_value.numpy()
      
            thv, filter_img = cv2.threshold(raw_image, 55, 255, cv2.THRESH_TOZERO)
            filtered_pos_matrix = np.multiply(pos_grads_matrix, filter_img)
            filtered_neg_matrix = np.multiply(neg_grads_matrix, filter_img)

            saliency_map = cv2.merge((neg_grads_matrix, raw_image, pos_grads_matrix))
            filtered_map = cv2.merge((filtered_neg_matrix, raw_image, filtered_pos_matrix)) 

            resized_saliency_map = cv2.resize(saliency_map,(1024,1024))
            resized_filtered_map = cv2.resize(filtered_map,(1024,1024))

            bordered_sal = cv2.copyMakeBorder(resized_saliency_map, top=4, bottom=4, left=4, right=2, borderType=cv2.BORDER_CONSTANT, value=255)
            bordered_fil = cv2.copyMakeBorder(resized_filtered_map, top=4, bottom=4, left=2, right=4, borderType=cv2.BORDER_CONSTANT, value=255)
            
            both = np.concatenate((bordered_sal, bordered_fil), axis=1)

            ind = str(feature_index + 1).zfill(nulls)

            cv2.imwrite(img_out_path + output_layer_name + '_image_[' + img_name + ']_feature_maps_' + str(value_scalar) + '_' + ind + '.png', both)

gap_saliency_map = FilterBackpropSaliencyMapping(model_load_path='/saved/model/directory/model.h5',
                                                 img_out_path='/mapped/to/images/activations/directory/',
                                                 images_dir='input/images/directory/',
                                                 input_layer_name='model_input_layer',
                                                 output_layer_name='model_gap_layer',
                                                 num_of_features=1024) #Number of the named gap layer filters.
