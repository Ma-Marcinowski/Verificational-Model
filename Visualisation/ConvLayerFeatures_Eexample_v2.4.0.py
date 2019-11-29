import tensorflow as tf
import numpy as np
import cv2

from tqdm import tqdm_notebook as tqdm

from tensorflow.keras.models import Model, load_model

tf.enable_eager_execution()

def InitializeImage():

    random_tensor2 = tf.keras.backend.random_uniform([256, 256], minval=0.0, maxval=1.0, dtype='float32')
    random_tensor3 = tf.keras.backend.expand_dims(random_tensor2, axis=-1)
    random_tensor4 = tf.keras.backend.expand_dims(random_tensor3, axis=0)

    return random_tensor4

def GradientAscent(image, filter_index, conv_layer, ascent_steps, ascent_rate):

    for step in tqdm(range(ascent_steps), desc='Filter ' + str(filter_index + 1) + ' features ascent:', leave=False):

        with tf.GradientTape() as tape:
          
            tape.watch(image)

            conv_layer_output = conv_layer(image)

            filter_activation = tf.image.total_variation(conv_layer_output[:, :, :, filter_index])
            
        filter_gradient = tape.gradient(filter_activation, image)

        normalized_gradient = tf.math.divide(filter_gradient, tf.math.add(tf.math.reduce_std(filter_gradient), 1e-9))

        image = tf.math.add(image, (tf.math.multiply(normalized_gradient, ascent_rate)))

    return image

def OctaveRescaling(image, filter_index, conv_layer, ascent_steps, ascent_rate, octave_rescales, octave_scale):

    for rescale in tqdm(range(octave_rescales), desc='Filter ' + str(filter_index + 1) + ' octave rescaling:', leave=False):

        shape = tf.cast(tf.shape(image[0, :, :, 0]), tf.float32)

        reshape = tf.cast(tf.math.multiply(shape, octave_scale), tf.int32)

        image = tf.image.resize(image, reshape)

        image = GradientAscent(image=image, 
                               filter_index=filter_index,
                               conv_layer=conv_layer,
                               ascent_steps=ascent_steps,
                               ascent_rate=ascent_rate)

    return image

def ConvLayerFeatures(model_load_path, img_out_path, 
                      input_layer_name, conv_layer_name, num_of_filters, 
                      ascent_steps, ascent_rate, 
                      octave_rescales, octave_scale):

    model = load_model(model_load_path)

    conv_layer = Model(inputs=model.get_layer(name=input_layer_name).input, outputs=model.get_layer(name=conv_layer_name).output)

    nulls = len(str(num_of_filters))

    for filter_index in tqdm(range(num_of_filters), desc=conv_layer_name + ' features visualisation:', leave=True):

        initial_image = InitializeImage()

        features_tensor = GradientAscent(image=initial_image, 
                                         filter_index=filter_index,
                                         conv_layer=conv_layer,
                                         ascent_steps=ascent_steps,
                                         ascent_rate=ascent_rate)

        rescaled_tensor = OctaveRescaling(image=features_tensor, 
                                          filter_index=filter_index,
                                          conv_layer=conv_layer,
                                          ascent_steps=ascent_steps,
                                          ascent_rate=ascent_rate,
                                          octave_rescales=octave_rescales, 
                                          octave_scale=octave_scale)

        img_tensor = rescaled_tensor.numpy()
        img_matrix = img_tensor[0, :, :, 0]
  
        normalized = cv2.normalize(img_matrix, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

        #blurred = cv2.GaussianBlur(normalized,(5, 5),0)
        #retv, ots = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        ind = str(filter_index + 1).zfill(nulls)

        cv2.imwrite(img_out_path + conv_layer_name + '_features_' + ind + '.png', normalized)
        
        #cv2.imwrite(img_out_path + conv_layer_name + '_features_' + ind + '.png', ots)

named_conv_layer_filter_features_visualised = ConvLayerFeatures(model_load_path='/saved/model/directory/model.h5',
                                                                img_out_path='/visualised/features/directory/',
                                                                input_layer_name='model_input_layer',
                                                                conv_layer_name='x_conv_layer_to_visualise',
                                                                num_of_filters=int_of_filters_on_the_named_layer,
                                                                ascent_steps=100,
                                                                ascent_rate=0.01,
                                                                octave_scale=1.1,
                                                                octave_rescales=10)
