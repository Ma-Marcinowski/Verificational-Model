import tensorflow as tf
import numpy as np
import cv2
import os

from tqdm import tqdm_notebook as tqdm

from tensorflow.keras.models import Model, load_model

tf.enable_eager_execution()

def InitializeStillImage(img_path):

    still_tensor2 = cv2.imread(img_path, 0).astype('float32')
    still_tensor3 = tf.keras.backend.expand_dims(still_tensor2, axis=-1)
    still_tensor4 = tf.keras.backend.expand_dims(still_tensor3, axis=0) / 255.

    return still_tensor4

def InitializeImage():

    random_tensor2 = tf.keras.backend.random_uniform([256, 256], minval=0.0, maxval=1.0, dtype='float32')
    random_tensor3 = tf.keras.backend.expand_dims(random_tensor2, axis=-1)
    random_tensor4 = tf.keras.backend.expand_dims(random_tensor3, axis=0)

    return random_tensor4

def GradientAscent(image, still, class_of_features, model, ascent_steps, ascent_rate):

    for step in tqdm(range(ascent_steps), desc='Output ' + class_of_features + ' features ascent:', leave=False):

        with tf.GradientTape() as tape:
          
            tape.watch(image)

            model_output = model([image, still])

            if class_of_features == 'positive':

              output_activation = model_output

            elif class_of_features == 'negative':

              output_activation = tf.math.multiply(model_output, -1.0)

            else:

              continue
                    
        output_gradient = tape.gradient(output_activation, image)

        normalized_gradient = tf.math.divide(output_gradient, tf.math.add(tf.math.reduce_std(output_gradient), 1e-9))

        image = tf.math.add(image, (tf.math.multiply(normalized_gradient, ascent_rate)))

    return output_activation, image

def OctaveRescaling(image, still, class_of_features, model, ascent_steps, ascent_rate, octave_rescales, octave_scale):

    for rescale in tqdm(range(octave_rescales), desc='Output ' + class_of_features + ' octave rescaling:', leave=False):

        shape = tf.cast(tf.shape(image[0, :, :, 0]), tf.float32)

        reshape = tf.cast(tf.math.multiply(shape, octave_scale), tf.int32)

        image = tf.image.resize(image, reshape)

        output_activation, image = GradientAscent(image=image,
                                   still=still,
                                   class_of_features=class_of_features,
                                   model=model,
                                   ascent_steps=ascent_steps,
                                   ascent_rate=ascent_rate)

    return output_activation, image

def ClassFeaturesVisualisation(model_load_path, img_out_path, images_dir,
                               left_input_name, right_input_name, output_layer_name,
                               class_of_features,
                               ascent_steps, ascent_rate, 
                               octave_rescales, octave_scale):

    model = load_model(model_load_path)

    model = Model(inputs=[model.get_layer(name=left_input_name).input, model.get_layer(name=right_input_name).input], outputs=model.get_layer(name=output_layer_name).output)

    image_names = os.listdir(images_dir)
    image_paths = [images_dir + n for n in image_names]
    images = zip(image_paths, image_names)

    initial_image = InitializeImage()

    for img_path, img_name in tqdm(images, total=len(image_paths), desc='Output ' + class_of_features + ' features mapping:', leave=True):

        still_image = InitializeStillImage(img_path=img_path)

        initial_activation, features_tensor = GradientAscent(image=initial_image,
                                                             still=still_image,
                                                             class_of_features=class_of_features,
                                                             model=model,
                                                             ascent_steps=ascent_steps,
                                                             ascent_rate=ascent_rate)

        final_activation, rescaled_tensor = OctaveRescaling(image=features_tensor,
                                                            still=still_image,
                                                            class_of_features=class_of_features,
                                                            model=model,
                                                            ascent_steps=ascent_steps,
                                                            ascent_rate=ascent_rate,
                                                            octave_rescales=octave_rescales, 
                                                            octave_scale=octave_scale)

        img_tensor = rescaled_tensor.numpy()
        img_matrix = img_tensor[0, :, :, 0]

        normalized_matrix = cv2.normalize(img_matrix, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
        still_image_matrix = cv2.imread(img_path, 0).astype(np.uint8)
        still_image_resized = cv2.resize(still_image_matrix, normalized_matrix.shape)

        class_left =  cv2.copyMakeBorder(normalized_matrix, top=4, bottom=4, left=2, right=4, borderType=cv2.BORDER_CONSTANT, value=255)
        still_right = cv2.copyMakeBorder(still_image_resized, top=4, bottom=4, left=4, right=2, borderType=cv2.BORDER_CONSTANT, value=255)

        both_imgs = np.concatenate((class_left, still_right), axis=1)

        activation_value = final_activation.numpy()

        cv2.imwrite(img_out_path + output_layer_name + '_' + class_of_features + '_features_[' + img_name + ']_' + str(activation_value) + '.png', both_imgs)

positive_class_features = ClassFeaturesVisualisation(model_load_path='/saved/model/directory/model.h5', 
                                                     img_out_path='/visualised/features/directory/',
                                                     images_dir='/input/images/directory',
                                                     left_input_name='model_left_input',
                                                     right_input_name='model_right_input',
                                                     output_layer_name='model_output_layer', #or one of the distance layers output
                                                     class_of_features='positive',
                                                     ascent_steps=100,
                                                     ascent_rate=0.01,
                                                     octave_scale=1.1,
                                                     octave_rescales=10)

negative_class_features = ClassFeaturesVisualisation(model_load_path='/saved/model/directory/model.h5',
                                                     img_out_path='/visualised/features/directory/',
                                                     images_dir='/input/images/directory',
                                                     left_input_name='model_left_input',
                                                     right_input_name='model_right_input',
                                                     output_layer_name='model_output_layer', #or one of the distance layers output
                                                     class_of_features='negative',
                                                     ascent_steps=100,
                                                     ascent_rate=0.01,
                                                     octave_scale=1.1,
                                                     octave_rescales=10)
  
