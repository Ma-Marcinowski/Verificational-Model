import os
import csv
import glob
import random
import sklearn
import numpy as np
import pandas as pd

from tqdm import tqdm

def Dataframe(img_path, df_path, df_img_path):

    os.chdir(img_path)

    pngs = glob.glob('*.png')

    with open(df_path, 'a+') as f:

        writer = csv.writer(f)

        rev = set()

        pos = 0

        for j in tqdm(pngs, leave=False):
            for i in pngs:

                if j[:8] == i[:8] and j[9] in ('7', '8') and i[9] in ('7', '8') and (j, i) not in rev:

                    pair = [df_img_path + j, df_img_path + i, 1]

                    writer.writerow(pair)

                    pos += 1

                    rev.add((i, j))

                else:

                    continue

        print('Done hard criterion positives: ', pos, ' instances.')

        neg = 0

        while neg < pos:

            j = random.choice(pngs)
            i = random.choice(pngs)

            if j[:8] != i[:8] and j[9] in ('7', '8') and i[9] in ('7', '8') and (j, i) not in rev:

                pair = [df_img_path + j, df_img_path + i, 0]

                writer.writerow(pair)

                neg += 1

                rev.add((i, j))

                print('%.2f%%'%(100*neg/pos), end="\r")

            else:

                continue

        else:

            print('Done hard criterion negatives: ', neg, ' instances.')

    df = pd.read_csv(df_path, header=None)

    df = sklearn.utils.shuffle(df)

    df.to_csv(df_path, header=['Leftname', 'Rightname', 'Label'], index=False)

    print('Done hard criterion dataframe: ', df.shape[0], ' image pairs.')

Hard_Criterion_Dataframe = Dataframe(img_path='/preprocessed/test/images/directory/',
                                     df_path='/dataframe/save/directory/Hard_Criterion_Dataframe.csv',
                                     df_img_path='/preprocessed/test/images/directory/indicated/in/the/test/dataframe')
