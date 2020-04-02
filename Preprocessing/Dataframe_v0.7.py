import glob
import os
import csv
import random
import sklearn
import numpy as np
import pandas as pd

from tqdm import tqdm

def Dataframe(mode, img_path, df_path, df_img_path, df_parts):

    os.chdir(img_path)

    pngs = glob.glob('*.png')

    with open(df_path, 'a+') as f:

        writer = csv.writer(f)

        rev = set()

        pos = 0

        for j in tqdm(pngs, leave=False):
            for i in pngs:

                if j[:8] == i[:8] and (j, i) not in rev:

                    pair = [df_img_path + j, df_img_path + i, 1]

                    writer.writerow(pair)

                    pos += 1

                    rev.add((i, j))

                else:

                    continue

        print('Done ' + mode + ' positives: ', pos, ' instances.')

        if mode == 'test' or mode == 'train':

            neg = 0

            for j in tqdm(pngs, leave=False):
                for i in pngs:

                    if j[:8] != i[:8] and (j, i) not in rev:

                        pair = [df_img_path + j, df_img_path + i, 0]

                        writer.writerow(pair)

                        neg += 1

                        rev.add((i, j))

                    else:

                        continue

            print('Done ' + mode + ' negatives: ', neg, ' instances.')

        elif mode == 'validation':

            neg = 0

            while neg < pos:

                j = random.choice(pngs)
                i = random.choice(pngs)

                if j[:8] != i[:8] and (j, i) not in rev:

                    pair = [df_img_path + j, df_img_path + i, 0]

                    writer.writerow(pair)

                    neg += 1

                    rev.add((i, j))

                    print('%.2f%%'%(100*neg/pos), end="\r")

                else:

                    continue

            else:

                print('Done ' + mode + ' negatives: ', neg, ' instances.')

    df = pd.read_csv(df_path, header=None)

    df = sklearn.utils.shuffle(df)

    df.to_csv(df_path, header=['Leftname', 'Rightname', 'Label'], index=False)

    print('Done ' + mode + ' dataframe: ', df.shape[0], ' image pairs.')

    if mode == 'train' or mode == 'validation' and df_parts != 0:

        df = pd.read_csv(df_path)

        pdf = np.split(df, df_parts, axis=0)

        for idx, p in enumerate(pdf, start=1):

            p.to_csv(df_path[:-4] + '-' + str(idx) + '.csv', header=['Leftname', 'Rightname', 'Label'], index=False)

            p.dropna(axis=0, how='any', inplace=True)

            print('Done train dataframe part ', idx, ': ', p.shape[0], ' image pairs.')

        print('Done splitting ' + mode + ' dataframes.')

TrainDataframe = Dataframe(mode='train',
                           img_path='/preprocessed/train/images/directory/',
                           df_path='/dataframe/save/directory/TrainDataframe.csv',
                           df_img_path='/preprocessed/train/images/directory/indicated/in/the/train/dataframe',
                           df_parts=6) #any number of partial train dataframes to generate

TestDataframe = Dataframe(mode='test',
                          img_path='/preprocessed/test/images/directory/',
                          df_path='/dataframe/save/directory/TestDataframe.csv',
                          df_img_path='/preprocessed/test/images/directory/indicated/in/the/test/dataframe/',
                          df_parts=0)

ValidDataframe = Dataframe(mode='validation',
                           img_path='/preprocessed/test/images/directory/',
                           df_path='/dataframe/save/directory/ValidDataframe.csv',
                           df_img_path='/preprocessed/test/images/directory/indicated/in/the/validation/dataframe',
                           df_parts=6) #any number of partial validation dataframes to generate
print('Dataframes done.')
