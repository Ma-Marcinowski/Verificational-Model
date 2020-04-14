import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import csv
import os

from tqdm.notebook import tqdm

from tensorflow.keras.models import Model, load_model

tf.enable_eager_execution()

def InitializeImage(img_path):

    tensor2 = cv2.imread(img_path, 0).astype('float32')
    tensor3 = tf.keras.backend.expand_dims(tensor2, axis=-1)
    tensor4 = tf.keras.backend.expand_dims(tensor3, axis=0) / 255.

    return tensor4

def AverageFeaturesDistribution(model_load_path, img_in_dir, df_out_path, input_layer_name, output_layer_name, dataset):

    model = load_model(model_load_path)

    conv_model = Model(inputs=model.get_layer(name=input_layer_name).input, outputs=model.get_layer(name=output_layer_name).output)

    features_vectors = []

    image_names = os.listdir(img_in_dir)
    image_paths = [img_in_dir + n for n in image_names]
    images = zip(image_paths, image_names)

    for img_path, img_name in tqdm(images, total=len(image_paths), desc='Averaging ' + dataset + ' features distributions', leave=True):

        if dataset in img_name: 

            image = InitializeImage(img_path=img_path)

            conv_model_output = conv_model(image)

            output_values = conv_model_output[0, :].numpy()

            features_vectors.append(output_values)

        else:

            continue

    num_of_vectors = len(features_vectors)

    sum_of_vectors = np.sum(features_vectors, axis=0)

    avr_of_vectors = np.divide(sum_of_vectors, num_of_vectors)

    df = pd.DataFrame(avr_of_vectors, index=None, columns=None)

    df.to_csv(df_out_path, index=False, header=None)

'''
def VisualisedFeaturesDistribution_One(iam_distro_df, cvl_distro_df, img_out_path):

    cvl_df = pd.read_csv(cvl_distro_df)
    iam_df = pd.read_csv(iam_distro_df)

    cvl_avr_distro = cvl_df.values.tolist()
    iam_avr_distro = iam_df.values.tolist()

    cvl_feature_indexes = list(range(len(cvl_avr_distro)))
    iam_feature_indexes = list(range(len(iam_avr_distro)))

    plt.rcParams['figure.figsize']=(200, 5)
    plt.plot(cvl_feature_indexes, cvl_avr_distro, '^m--', )
    plt.plot(iam_feature_indexes, iam_avr_distro, 'ob:', )
    plt.xticks(np.arange(0, 1024, 1), rotation=90)
    plt.grid(b=True, axis='both')
    plt.title('Database Features Distributions')
    plt.ylabel('Average Feature Value')
    plt.xlabel('Feature Index')
    plt.legend(['CVL', 'IAM'], loc='upper left')
    plt.savefig(fname=img_out_path, dpi=150)
    plt.show()
    
def VisualisedFeaturesDistribution_Two(iam_distro_df, cvl_distro_df, img_out_path):

    cvl_df = pd.read_csv(cvl_distro_df)
    iam_df = pd.read_csv(iam_distro_df)

    cvl_avr_distro = cvl_df.values.tolist()
    iam_avr_distro = iam_df.values.tolist()

    cvl_feature_indexes = list(range(len(cvl_avr_distro)))
    iam_feature_indexes = list(range(len(iam_avr_distro)))

    listed = [cvl_avr_distro, cvl_feature_indexes, iam_avr_distro, iam_feature_indexes]
    listed_sorted = sort_together(iterables=listed, key_list=(0, ), reverse=False)

    normalized_cvl = normalize(listed_sorted[0], norm='l1', axis=0)
    normalized_iam = normalize(listed_sorted[2], norm='l1', axis=0)

    plt.rcParams['figure.figsize']=(200, 5)
    plt.plot(cvl_feature_indexes, normalized_cvl, '^m--', )
    plt.plot(iam_feature_indexes, normalized_iam, 'ob:', )
    plt.xticks(np.arange(0, 1024, 1), labels=listed_sorted[1], rotation=90)
    plt.grid(b=True, axis='x')
    plt.title('Database Features Distributions')
    plt.ylabel('Average Feature Value')
    plt.xlabel('Feature Index')
    plt.legend(['CVL', 'IAM'], loc='upper left')
    plt.savefig(fname=img_out_path, dpi=150)
    plt.show()
'''

def CompareDistributions(iam_distro_df, cvl_distro_df, filters_to_remove_df):

    cvl_df = pd.read_csv(cvl_distro_df)
    iam_df = pd.read_csv(iam_distro_df)

    cvl_avr_distro = cvl_df.values.tolist()
    iam_avr_distro = iam_df.values.tolist()

    feature_indexes = list(range(len(cvl_avr_distro)))

    averages = zip(cvl_avr_distro, iam_avr_distro, feature_indexes)

    inequivalent = []

    for cvl_avr, iam_avr, feature_index in tqdm(averages, total=len(feature_indexes), desc='Comparing features distributions', leave=True):


        if np.absolute(np.subtract(cvl_avr, iam_avr)) >= 0.25:

            inequivalent.append(feature_index)

        else:

            continue

    df = pd.DataFrame(inequivalent, index=None, columns=None)

    df.to_csv(filters_to_remove_df, index=False, header=None)
    
def RemoveFilters(model_load_path, modified_model_save_path, left_layer_name, right_layer_name, filters_to_remove_df):

    model = load_model(model_load_path)

    left_conv_layer = model.get_layer(name=left_layer_name)
    left_conv_layer_weights = left_conv_layer.get_weights()
    
    right_conv_layer = model.get_layer(name=right_layer_name)
    right_conv_layer_weights = right_conv_layer.get_weights()

    df = pd.read_csv(filters_to_remove_df)
    filters_to_remove = df.values.tolist()

    for filter_index in tqdm(filters_to_remove, total=len(filters_to_remove), desc='Removing ' + str(len(filters_to_remove)) + ' filters', leave=True):

        zeroed_mat = np.zeros_like(left_conv_layer_weights[0][:, :, :, filter_index])
        zeroed_bias = np.zeros_like(left_conv_layer_weights[1][filter_index])

        left_conv_layer_weights[0][:, :, :, filter_index] = zeroed_mat
        left_conv_layer_weights[1][filter_index] = zeroed_bias

        right_conv_layer_weights[0][:, :, :, filter_index] = zeroed_mat
        right_conv_layer_weights[1][filter_index] = zeroed_bias

    left_conv_layer.set_weights(left_conv_layer_weights)
    right_conv_layer.set_weights(right_conv_layer_weights)

    model.save(modified_model_save_path, overwrite=True, include_optimizer=False)

    print('Modified model saved.')

cvl = AverageFeaturesDistribution(model_load_path='/saved/model/directory/model.h5', 
                                  img_in_dir='/test/images/directory/', 
                                  df_out_path='/path/to/Average_Features_Distribution_CVL.csv',
                                  input_layer_name='model_input', 
                                  output_layer_name='model_features_output',
                                  dataset='cvl')

iam = AverageFeaturesDistribution(model_load_path='/saved/model/directory/model.h5', 
                                  img_in_dir='/test/images/directory/', 
                                  df_out_path='/path/to/Average_Features_Distribution_IAM.csv', 
                                  input_layer_name='model_input', 
                                  output_layer_name='model_features_output',
                                  dataset='iam')

'''
image_1 = VisualisedFeaturesDistribution_One(iam_distro_df='/path/to/Average_Features_Distribution_CVL.csv', 
                                             cvl_distro_df='/path/to/Average_Features_Distribution_IAM.csv', 
                                             img_out_path='/path/to/Visualised_Features_Distribution_Image.png')

image_2 = VisualisedFeaturesDistribution_Two(iam_distro_df='/path/to/Average_Features_Distribution_CVL.csv', 
                                             cvl_distro_df='/path/to/Average_Features_Distribution_IAM.csv', 
                                             img_out_path='/path/to/Visualised_Features_Distribution_Image.png')
'''

inequivalent = CompareDistributions(cvl_distro_df='/path/to/Average_Features_Distribution_CVL.csv',,
                                    iam_distro_df='/path/to/Average_Features_Distribution_IAM.csv', 
                                    filters_to_remove_df='/path/to/Filters_to_Remove_Dataframe.csv')

modified_model = = RemoveFilters(model_load_path='/saved/model/directory/model.h5', 
                                 modified_model_save_path='/modified/model/save/directory/modified_model.h5', 
                                 left_layer_name='3rdConvLeft',
                                 right_layer_name='3rdConvRight',
                                 filters_to_remove_df='/path/to/Filters_to_Remove_Dataframe.csv')
