import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def Preprocessing(mode, in_path, out_path, filters_dir):

    k_names = os.listdir(filters_dir)
    kernels = [filters_dir + n for n in k_names]

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

        th, den = cv2.threshold(inv, 15, 255, cv2.THRESH_TOZERO)

        y=730
        x=230
        h=2048
        w=2048

        cropped = den[y:y+h, x:x+w]

        resized = cv2.resize(cropped,(1024,1024))

        horizontal = np.split(resized, 4, axis=1)

        for idx, h in enumerate(horizontal, start=1):

            vertical = np.split(h, 4, axis=0)

            for ind, v in enumerate(vertical, start=1):

                thv, denv = cv2.threshold(v, 55, 255, cv2.THRESH_TOZERO)

                filtered = []

                for kern in kernels:

                    kernel = cv2.imread(kern, 0)

                    ksum = np.sum(np.multiply(denv, kernel))

                    filtered.append(ksum)

                if 0 not in filtered:

                    vc = v.copy()

                    noise = cv2.randn(vc, 0, 15)

                    vn = v + noise

                    cv2.imwrite(out_path + str(i) + '-' + str(f) + '-' + str(idx) + str(ind) + '.png', vn)

                else:

                    continue

    print(mode + ' preprocessing done: 100%')

TrainPreprocessing = Preprocessing(mode='train',
                                   in_path='/IAM/dataframe/of/raw/train/images/directories.csv',
                                   out_path='/preprocessed/train/images/save/directory/',
                                   filters_dir='/filters/directory/')

TestPreprocessing = Preprocessing(mode='test',
                                  in_path='/IAM/dataframe/of/raw/test/images/directories.csv',
                                  out_path='/preprocessed/test/images/save/directory/',
                                  filters_dir='/filters/directory/')

print('Preprocessing done.')
