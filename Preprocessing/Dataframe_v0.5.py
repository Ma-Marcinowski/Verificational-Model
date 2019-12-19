import glob
import os
import csv
import random
import sklearn
import numpy as np
import pandas as pd
from tqdm import tqdm

def Dataframe(mode, img_path, df_path, df_img_path, num_of_train_dfs, valid_df_path, valid_fraction):

    os.chdir(img_path)

    pngs = glob.glob('*.png')

    with open(df_path, 'a+') as f:

        writer = csv.writer(f)

        pos = 0

        for j in tqdm(pngs, leave=False):
            for i in pngs:

                if j[:8] == i[:8] and j[0] == i[0]:

                    pair = [df_img_path + j, df_img_path + i, 1]

                    writer.writerow(pair)

                    pos += 1

                else:

                    continue

        print('Done ' + mode + ' positives: ', pos, ' instances.')

        neg = 0

        while neg < pos:

            j = random.choice(pngs)
            i = random.choice(pngs)

            if j[:8] != i[:8] and j[0] == i[0]:

                pair = [df_img_path + j, df_img_path + i, 0]

                writer.writerow(pair)

                neg += 1

                print('%.2f%%'%(100*neg/pos), end="\r")

            else:

                continue

        else:

            print('Done ' + mode + ' negatives: ', neg, ' instances.')

    df = pd.read_csv(df_path, header=None)

    df = sklearn.utils.shuffle(df)

    df.to_csv(df_path, header=['Leftname', 'Rightname', 'Label'], index=False)

    print('Done ' + mode + ' dataframe: ', df.shape[0], ' image pairs.')

    if mode == 'train' and num_of_train_dfs != None:

        df = pd.read_csv(df_path)

        n = num_of_train_dfs

        pdf = np.split(df, n, axis=0)

        for idx, p in enumerate(pdf, start=1):

            p.to_csv(df_path[:-4] + '-' + str(idx) + '.csv', header=['Leftname', 'Rightname', 'Label'], index=False)

            p.dropna(axis=0, how='any', inplace=True)

            print('Done train dataframe part ', idx, ': ', p.shape[0], ' image pairs.')

        print('Done splitting train dataframes.')

    elif mode == 'test' and valid_fraction != None:

        tedf = pd.read_csv(df_path)

        vadf = tedf.sample(frac=valid_fraction, axis=0)

        vadf.to_csv(valid_df_path, index=False)

        print('Done validation dataframe: ', vadf.shape[0], ' image pairs.')

TrainDataframe = Dataframe(mode='train',
                           img_path='/preprocessed/train/images/directory/',
                           df_path='/dataframe/save/directory/TrainDataframe.csv',
                           df_img_path='/preprocessed/train/images/directory/indicated/in/the/train/dataframe',
                           num_of_train_dfs=2, #any even number of partial train dataframes to generate
                           valid_df_path=None,
                           valid_fraction=None)

TestDataframe = Dataframe(mode='test',
                          img_path='/preprocessed/test/images/directory/',
                          df_path='/dataframe/save/directory/TestDataframe.csv',
                          df_img_path='/preprocessed/test/images/directory/indicated/in/the/test/dataframe',
                          num_of_train_dfs=None,
                          valid_df_path='/dataframe/save/directory/ValidDataframe.csv',
                          valid_fraction=0.2) #any float of instances to be pulled from the test dataframe

print('Dataframes done.')
