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

        n = 200000
        neg = 0

        while neg < n:

            j = random.choice(pngs)
            i = random.choice(pngs)

            if j[:8] != i[:8] and j[0] in ('c') and i[0] in ('i') and (j, i) not in rev:

                pair = [df_img_path + j, df_img_path + i, 0]

                writer.writerow(pair)

                neg += 1

                rev.add((i, j))

                print('%.2f%%'%(100*neg/n), end="\r")

            else:

                continue

        else:

            print('Done negative criterion negatives: ', neg, ' instances.')

    df = pd.read_csv(df_path, header=None)

    df = sklearn.utils.shuffle(df)

    df.to_csv(df_path, header=['Leftname', 'Rightname', 'Label'], index=False)

    print('Done Negative Criterion dataframe: ', df.shape[0], ' image pairs.')

Negative_Criterion_Dataframe = Dataframe(img_path='/preprocessed/test/images/directory/',
                                         df_path='/dataframe/save/directory/Negative_Criterion_Dataframe.csv',
                                         df_img_path='/preprocessed/test/images/directory/indicated/in/the/test/dataframe')
