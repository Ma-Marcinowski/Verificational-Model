import os
import csv
import random
import sklearn
import numpy as np
import pandas as pd

from tqdm import tqdm

def Dataframe(mode, imgs_dir_512, imgs_dir_256, df_path, df_img_512_path, df_img_256_path, df_parts):

    names_512 = os.listdir(imgs_dir_512)
    names_256 = os.listdir(imgs_dir_256)

    with open(df_path, 'a+') as f:

        writer = csv.writer(f)

        rev = set()

        pos = 0

        for j in tqdm(names_512, leave=False):
            for i in names_256:

                if j[:8] == i[:8] and j[0] == i[0] and (j, i) not in rev:

                    pair = [df_img_512_path + j, df_img_256_path + i, 1]

                    writer.writerow(pair)

                    pos += 1

                    rev.add((i, j))

                else:

                    continue

        print('Done ' + mode + ' positives: ', pos, ' instances.')

        if mode == 'train' or mode == 'validation':

            neg = 0

            for neg in tqdm(range(pos), leave=False):

                j = random.choice(names_512)
                i = random.choice(names_256)

                if j[:8] != i[:8] and j[0] == i[0] and (j, i) not in rev:

                    pair = [df_img_512_path + j, df_img_256_path + i, 0]

                    writer.writerow(pair)

                    neg += 1

                    rev.add((i, j))

                else:

                    continue

            else:

                print('Done ' + mode + ' negatives: ', neg, ' instances.')

        elif mode == 'test':

            neg = 0

            for j in tqdm(names_512, leave=False):
                for i in names_256:

                    if j[:8] != i[:8] and j[0] == i[0] and (j, i) not in rev:

                        pair = [df_img_512_path + j, df_img_256_path + i, 0]

                        writer.writerow(pair)

                        neg += 1

                        rev.add((i, j))

                    else:

                        continue

            print('Done ' + mode + ' negatives: ', neg, ' instances.')

    df = pd.read_csv(df_path, header=None)

    df = sklearn.utils.shuffle(df)

    df.to_csv(df_path, header=['Leftname', 'Rightname', 'Label'], index=False)

    print('Done ' + mode + ' dataframe: ', df.shape[0], ' image pairs.')

    if (mode == 'train' or mode == 'validation') and df_parts != None:

        df = pd.read_csv(df_path)

        pdf = np.split(df, df_parts, axis=0)

        for idx, p in enumerate(pdf, start=1):

            p.to_csv(df_path[:-4] + '-' + str(idx) + '.csv', header=['Leftname', 'Rightname', 'Label'], index=False)

            p.dropna(axis=0, how='any', inplace=True)

            print('Done ' + mode + ' dataframe part ', idx, ': ', p.shape[0], ' image pairs.')

        print('Done splitting ' + mode + ' dataframes.')

TrainDataframe = Dataframe(mode='train',
                           imgs_dir_512='/preprocessed/512x512/train/images/directory/',
                           imgs_dir_256='/preprocessed/256x256/train/images/directory/',
                           df_path='/dataframe/save/directory/TrainDataframe.csv',
                           df_img_512_path='/preprocessed/512x512/train/images/directory/indicated/in/the/train/dataframe',
                           df_img_256_path='/preprocessed/256x256/train/images/directory/indicated/in/the/train/dataframe',
                           df_parts=None) #any number of partial train dataframes to generate

TestDataframe = Dataframe(mode='test',
                          imgs_dir_512='/preprocessed/512x512/test/images/directory/',
                          imgs_dir_256='/preprocessed/256x256/test/images/directory/',
                          df_path='/dataframe/save/directory/TrainDataframe.csv',
                          df_img_512_path='/preprocessed/512x512/test/images/directory/indicated/in/the/train/dataframe',
                          df_img_256_path='/preprocessed/256x256/test/images/directory/indicated/in/the/train/dataframe',
                          df_parts=None)

ValidDataframe = Dataframe(mode='validation',
                           imgs_dir_512='/preprocessed/512x512/test/images/directory/',
                           imgs_dir_256='/preprocessed/256x256/test/images/directory/',
                           df_path='/dataframe/save/directory/TrainDataframe.csv',
                           df_img_512_path='/preprocessed/512x512/test/images/directory/indicated/in/the/train/dataframe',
                           df_img_256_path='/preprocessed/256x256/test/images/directory/indicated/in/the/train/dataframe',
                           df_parts=None) #any number of partial train dataframes to generate

print('Dataframes done.')
