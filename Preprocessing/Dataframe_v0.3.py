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

        for j in tqdm(pngs, leave=False):
            for i in pngs:

                if j[:8] == i[:8]:

                    pair = [df_img_path + j, df_img_path + i, 1]

                    writer.writerow(pair)

                else:

                    continue

        print('Done ' + mode + ' positives: 100%')

        g = f.tell()
        k = 2 * f.tell()

        while g < k:

            j = random.choice(pngs)
            i = random.choice(pngs)

            if j[:8] != i[:8]:

                pair = [df_img_path + j, df_img_path + i, 0]

                writer.writerow(pair)

                g = f.tell()

                print('%.2f%%'%(100*g/k), end="\r")

            else:

                continue

        else:

            print('Done ' + mode + ' negatives: 100%')

    df = pd.read_csv(df_path, header=None)

    df = sklearn.utils.shuffle(df)

    df.to_csv(df_path, header=['Leftname', 'Rightname', 'Label'], index=False)

    print('Done ' + mode + ' dataframe.')

    if mode == 'train':

        df = pd.read_csv(df_path, skiprows=[0])

        n = num_of_train_dfs

        pdf = np.split(df, n, axis=0)

        for idx, p in enumerate(pdf, start=1):

            p.to_csv(df_path[:-4] + '-' + str(idx) + '.csv', header=['Leftname', 'Rightname', 'Label'], index=False)
            
            p.dropna(axis=0, how='any', inplace=True)

            print('Done train dataframe part: ', str(idx))

        print('Done splitting train dataframes.')

    elif mode == 'test':

        tedf = pd.read_csv(df_path)

        vadf = tedf.sample(frac=valid_fraction, axis=0)

        vadf.to_csv(valid_df_path, index=False)

        print('Done validation dataframe.')

TrainDataframe = Dataframe(mode='train',
                           img_path='/preprocessed/train/images/directory/',
                           df_path='/dataframe/save/directory/TrainDataframe.csv',
                           df_img_path='/preprocessed/train/images/directory/indicated/in/the/train/dataframe',
                           num_of_train_dfs=2, #any number of partial train dataframes to generate
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
