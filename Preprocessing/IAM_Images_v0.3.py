import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def Preprocessing(mode, in_path, out_path):

    InitDf = pd.read_csv(in_path)

    path = InitDf[mode+'Path'].tolist()
    ids = InitDf[mode+'ID'].tolist()
    form = InitDf[mode+'Form'].tolist()

    k = 0

    for j, i, f in zip(path, ids, form):

        k += 1
        print('%.2f%%'%(100*k/InitDf.shape[0]), end="\r")

        img = cv2.imread(j, 0)
        inv = np.bitwise_not(img)
        th, den = cv2.threshold(inv, 55, 255, cv2.THRESH_TOZERO)

        y=730
        x=230
        h=2048
        w=2048

        cropped = den[y:y+h, x:x+w]

        resized = cv2.resize(cropped,(1024,1024))

        blur = cv2.GaussianBlur(resized,(5,5),0)

        retv, ots = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        horizontal = np.split(ots, 4, axis=1)

        for idx, h in enumerate(horizontal, start=1):

            vertical = np.split(h, 4, axis=0)

            for ind, v in enumerate(vertical, start=1):

                mean = v.mean()

                if mean >= 8:

                    cv2.imwrite(out_path + str(i) + '-' + str(f) + '-' + str(idx) + str(ind) + '.png', v)

                else:

                    continue

    print(mode + ' preprocessing done: 100%')

TrainPreprocessing = Preprocessing(mode='train',
                                   in_path='/IAM/dataframe/of/raw/train/images.csv',
                                   out_path='/preprocessed/train/images/save/directory/')

TestPreprocessing = Preprocessing(mode='test',
                                  in_path='/IAM/dataframe/of/raw/test/images.csv',
                                  out_path='/preprocessed/test/images/save/directory/')

print('Preprocessing done.')
