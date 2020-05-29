import tensorflow as tf
import numpy as np
import pandas as pd
import random
import csv

from tqdm import tqdm_notebook as tqdm
from scipy.stats import mode

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense
from tensorflow.keras.layers import BatchNormalization, GaussianDropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_image(img):
    return img_to_array(load_img(img, color_mode='grayscale')) / 255.

class DataSequence(tf.keras.utils.Sequence):

    def __init__(self, dataframe, batch_size):
        self.df = pd.read_csv(dataframe)
        self.batch_size = batch_size

        self.labels = self.df['Label'].tolist()
        self.leftnames = self.df['Leftname'].tolist()
        self.rightnames = self.df['Rightname'].tolist()

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def get_batch_labels(self, idx):
        batch_labels = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        return np.array(batch_labels)

    def get_batch_leftnames(self, idx):
        batch_leftnames = self.leftnames[idx * self.batch_size: (idx + 1) * self.batch_size]
        return np.array([load_image(i) for i in batch_leftnames])

    def get_batch_rightnames(self, idx):
        batch_rightnames = self.rightnames[idx * self.batch_size: (idx + 1) * self.batch_size]
        return np.array([load_image(j) for j in batch_rightnames])

    def __getitem__(self, idx):
        batch_x1 = self.get_batch_leftnames(idx)
        batch_x2 = self.get_batch_rightnames(idx)
        batch_y = self.get_batch_labels(idx)
        return ({'left_input': batch_x1, 'right_input': batch_x2}, {'output': batch_y})

BatchSize = 128

TestSeq = DataSequence(dataframe='/path/TestDataframe.csv', batch_size = BatchSize)

model = load_model('/path/VM_v2.5.1.h5')

Adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False)

model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=['accuracy'])

predictions = model.predict_generator(generator=TestSeq,
                                      steps=None,
                                      callbacks=None,
                                      max_queue_size=10,
                                      workers=1,
                                      use_multiprocessing=False,
                                      verbose=1)

rounded_predictions = np.round(predictions, 4)

predicted = pd.DataFrame(rounded_predictions)

predicted.to_csv('/path/Predictions.csv', header=['Predictions'], index=False)

data_df = pd.read_csv('/path/TestSheet.csv')
pred_df = pd.read_csv('/path/Predictions.csv')

result = pd.concat(objs=[data_df, pred_df], axis=1, sort=False)

result.to_csv('/path/Partial_Results.csv', header=['Leftname', 'Rightname', 'Label', 'Prediction'], index=False)

def Specified_Results(partial_results_df, full_results_df):

    df = pd.read_csv(partial_results_df)

    lefts = df['Leftname'].tolist()
    rights = df['Rightname'].tolist()
    labels = df['Label'].tolist()
    predictions = df['Prediction'].tolist()

    df['LeftAuthor'] = df['Leftname'].str[:31]
    df['RightAuthor'] = df['Rightname'].str[:31]
    
    left_id = df['LeftAuthor'].tolist()
    right_id = df['RightAuthor'].tolist()

    listed = zip(left_id, right_id, lefts, rights, labels, predictions)

    with open(full_results_df, 'a+') as f:

      writer = csv.writer(f)

      for auth_l, auth_r, path_l, path_r, label, prediction in tqdm(listed, desc='Specifying results', leave=False):

        if prediction > 0.5 and label == 1:

            pair = [auth_l, auth_r, path_l, path_r, label, prediction, 1]

        elif prediction <= 0.5 and label == 0:

            pair = [auth_l, auth_r, path_l, path_r, label, prediction, 1]

        else:

            pair = [auth_l, auth_r, path_l, path_r, label, prediction, 0]

        writer.writerow(pair)

      print('Results specified: 100%')

    res = pd.read_csv(full_results_df, header=None)

    res.to_csv(full_results_df, header=['LeftAuthor', 'RightAuthor', 'Leftname', 'Rightname', 'Label', 'Prediction', 'Result'], index=False)

def Given_Acc_Probability(full_results_df, authors_sample, expected_acc_min, expected_acc_max, combinations_limit):

    df = pd.read_csv(full_results_df)

    lefts = df['LeftAuthor'].tolist()
    rights = df['RightAuthor'].tolist()
    results = df['Result'].tolist()

    listed = tuple(zip(lefts, rights, results)) #Zip is turned into a tuple, so it won't be exhausted by the outer loop's first step.

    authors_ids = list(set(lefts))

    accs = set()

    for step in tqdm(range(combinations_limit), desc='Evaluating samples', leave=True):

        sample = random.choices(authors_ids, k=authors_sample)
        
        corrects = 0
        incorrects = 0

        for left_id, right_id, result in listed:

              if result == 1 and left_id in sample and right_id in sample:

                  corrects += 1

              elif result == 0 and left_id in sample and right_id in sample:

                  incorrects += 1

              else:

                continue

        if (corrects + incorrects) > 0:

            acc = corrects / (corrects + incorrects)

            accs.add(acc)

        else:

            continue

    known_acc_events = len(accs)
    expected_acc_events = 0

    for acc_event in accs:

        if acc_event >= expected_acc_min and acc_event <= expected_acc_max:

            expected_acc_events += 1
        
        else:

           continue

    sample_acc_probability = round(expected_acc_events / known_acc_events, 4)

    round_acc_events = [round(a*100, 0) for a in accs ]
    mode_array, most_acc_events = mode(round_acc_events, axis=0, nan_policy='omit')

    dominant_acc_probability = round(most_acc_events / known_acc_events, 4) 

    print('Authors sample: ', authors_sample)
    print('Acc lower range: ', expected_acc_min)
    print('Acc upper range: ', expected_acc_max)
    print('The probability for the Acc observed is: ', sample_acc_probability)
    print('The most randomly observed Acc is: ', (mode_array[0] / 100))
    print('The probability for the most observed Acc is: ', dominant_acc_probability)

results = Specified_Results(partial_results_df='/path/Partial_Results.csv',
                            full_results_df='/path/Specified_Results.csv')

probability = Given_Acc_Probability(full_results_df='/path/Specified_Results.csv',
                                    authors_sample=10, #any number of authors to sample
                                    expected_acc_min=0.9100, #any lower range of the expected acc
                                    expected_acc_max=0.9200, #any upper range of the expected acc
                                    combinations_limit=1000) #any limit of random combinations to utilize
