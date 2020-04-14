import tensorflow as tf
import pandas as pd
import numpy as np
import scipy
import cv2
import csv
import os

from tqdm import tqdm_notebook as tqdm

from sklearn.preprocessing import normalize

from tensorflow.keras.models import Model, load_model

tf.enable_eager_execution()

def InitializeImage(img_path):

    tensor2 = cv2.imread(img_path, 0).astype('float32')
    tensor3 = tf.keras.backend.expand_dims(tensor2, axis=-1)
    tensor4 = tf.keras.backend.expand_dims(tensor3, axis=0) / 255.

    return tensor4

def AggregateFeaturesDistributions(model_load_path, img_in_dir, df_out_path, input_layer_name, output_layer_name):

    model = load_model(model_load_path)

    conv_model = Model(inputs=model.get_layer(name=input_layer_name).input, outputs=model.get_layer(name=output_layer_name).output)

    features_vectors = []

    image_names = os.listdir(img_in_dir)
    image_paths = [img_in_dir + n for n in image_names]
    images = zip(image_paths, image_names)

    for img_path, img_name in tqdm(images, total=len(image_paths), desc='Aggregating features distributions', leave=True):

        image = InitializeImage(img_path=img_path)

        conv_model_output = conv_model(image)

        output_values = list(conv_model_output[0, :].numpy())

        output_values.insert(0, img_name)

        features_vectors.append(output_values)
        
    df = pd.DataFrame(features_vectors, index=None, columns=None)

    df.to_csv(df_out_path, index=False, header=None)
    
def EquivalentFeaturesDistributions(extracts_distro_df, patches_distro_df, equiv_distro_df):

    edf = pd.read_csv(extracts_distro_df)
    pdf = pd.read_csv(patches_distro_df)

    extracts = edf.values.tolist()
    patches = pdf.values.tolist()

    with open(equiv_distro_df, 'a+') as f:
        
        writer = csv.writer(f)

        for extract in tqdm(extracts, total=len(extracts), desc='Searching for equivalent distributions', leave=True):

            for patch in tqdm(patches, total=len(patches), desc='inner-loop', leave=False):

                ext_name = extract[0]
                pch_name = patch[0]

                if ext_name[:8] == pch_name[:8]:

                    extract_features = extract[1:]
                    patch_features = patch[1:]

                    reshaped_extract = np.array(extract_features).reshape(1024, 1)
                    reshaped_patch = np.array(patch_features).reshape(1024, 1)

                    normalized_extract = normalize(reshaped_extract, norm='l1', axis=0)
                    normalized_patch = normalize(reshaped_patch, norm='l1', axis=0)

                    cosine_distance = scipy.spatial.distance.cosine(normalized_extract, normalized_patch)

                    if cosine_distance <= 0.05:

                        writer.writerow([extract[0], patch[0], str(cosine_distance)])

                    else:

                        continue

                else:

                    continue

    df = pd.read_csv(equiv_distro_df, header=None)

    df.to_csv(equiv_distro_df, header=['Extract', 'Patch', 'Difference'], index=False)
    
extracts = AggregateFeaturesDistributions(model_load_path='/saved/model/directory/model.h5', 
                                         img_in_dir='/test/image/extracts/directory/', 
                                         df_out_path='/path/to/Extracts_Features_Distributions.csv', 
                                         input_layer_name='left_input', 
                                         output_layer_name='LeftGAP')

patches = AggregateFeaturesDistributions(model_load_path='/saved/model/directory/model.h5', 
                                         img_in_dir='/test/image/patches/directory/', 
                                         df_out_path='/path/to/Patches_Features_Distributions.csv', 
                                         input_layer_name='left_input', 
                                         output_layer_name='LeftGAP')

equivalent = EquivalentFeaturesDistributions(extracts_distro_df='/path/to/Extracts_Features_Distributions.csv',
                                             patches_distro_df='/path/to/Patches_Features_Distributions.csv',
                                             equiv_distro_df='/path/to/Distributions_Equivalent_Patches_and_Extracts_Dataframe.csv')
