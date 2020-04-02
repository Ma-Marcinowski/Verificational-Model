import tensorflow as tf
import numpy as np
import pandas as pd
import csv

from tqdm import tqdm_notebook as tqdm

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

model = load_model('/path/VM_v3.6.0.h5')

Adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False)

model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=['accuracy'])

predictions = model.predict(x=TestSeq,
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

def Combined_Evaluation(partial_results_df, full_results_df, evaluation_df):

    df = pd.read_csv(partial_results_df)

    df['LeftImg'] = df['Leftname'].str[:-7]
    df['RightImg'] = df['Rightname'].str[:-7]

    df.to_csv(partial_results_df, header=['Leftname', 'Rightname', 'Label', 'Prediction', 'LeftImg', 'RightImg'], index=False)

    df = pd.read_csv(partial_results_df)

    lefts = df['LeftImg'].tolist()
    rights = df['RightImg'].tolist()

    lrs = set(zip(lefts, rights))

    df.set_index(['LeftImg', 'RightImg'], inplace=True)

    with open(full_results_df, 'a+') as f:

      writer = csv.writer(f)

      for l, r in tqdm(lrs, desc='Results accumulation:', leave=False):

        sdf = df.xs(key=(l, r), axis=0, level=['LeftImg', 'RightImg'])

        predictions = sdf['Prediction'].tolist()

        label = sdf['Label'].iloc[0]

        pos = 0
        neg = 0
        unr = 0

        for p in predictions:

          if p > 0.75:

            pos += 1

          elif p < 0.25:

            neg +=1

          else:

            unr +=1

        if pos > (neg + unr):

          res = 1

        elif neg > (pos + unr):

          res = 0

        else:

          res = 0.5

        pair = [l, r, label, pos, neg, unr, res]

        writer.writerow(pair)

      print('Done accumulation: 100%')

    res = pd.read_csv(full_results_df, header=None)

    res.to_csv(full_results_df, header=['LeftImg', 'RightImg', 'Label', 'Positives', 'Negatives', 'Unresolves', 'Result'], index=False)

    df = pd.read_csv(full_results_df)

    labels = df['Label'].tolist()
    results = df['Result'].tolist()

    with open(evaluation_df, 'a+') as f:

      writer = csv.writer(f)

      s = len(labels)
      t = 0

      TP = 1
      TN = 1
      FP = 1
      FN = 1

      UNS = 1

      for l, r in zip(labels, results):

        if l == 1 and r == 1:

          TP += 1

        elif l == 0 and r == 0:

          TN += 1

        elif l == 0 and r == 1:

          FP += 1

        elif l == 1 and r == 0:

          FN +=1

        else:

          UNS += 1

        t += 1

        print('%.2f%%'%(100*t/s), end="\r")

      print('Done evaluation: 100%')

      Acc = round((TP + TN) / (TP + TN + FP + FN), 4)
      TPR = round(TP / (TP + FN), 4)
      TNR = round(TN / (TN + FP), 4)
      FPR = round(FP / (FP + TN), 4)
      FNR = round(FN / (FN + TP), 4)
      PPV = round(TP / (TP + FP), 4)
      NPV = round(TN / (TN + FN), 4)
      Unresolved = round(UNS / (TP + TN + FP + FN + UNS), 4)

      evaluations = [Acc, TPR, TNR, FPR, FNR, PPV, NPV, Unresolved]

      writer.writerow(evaluations)

    ev = pd.read_csv(evaluation_df, header=None)

    ev.to_csv(evaluation_df, header=['Acc', 'TPR', 'TNR', 'FPR', 'FNR', 'PPV', 'NPV', 'Unresolved'], index=False)

    print('Done combined evaluation.')

    print('Acc = ', Acc)
    print('TPR = ', TPR)
    print('TNR = ', TNR)
    print('FPR = ', FPR)
    print('FNR = ', FNR)
    print('PPV = ', PPV)
    print('NPV = ', NPV)
    print('Unresolved = ', Unresolved)

evaluated = Combined_Evaluation(partial_results_df='/path/Partial_Results.csv',
                                full_results_df='/path/Results.csv',
                                evaluation_df='/path/Evaluation.csv')
