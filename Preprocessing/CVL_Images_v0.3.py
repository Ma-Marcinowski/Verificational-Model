import glob
import os
import cv2
import numpy as np
from tqdm import tqdm

def Preprocessing(mode, in_path, out_path):

    os.chdir(in_path)

    tifs = glob.glob('*.tif')

    for j in tqdm(tifs, desc=mode+'-loop'):

        img = cv2.imread(j, 0)

        blur = cv2.GaussianBlur(img,(5,5),0)

        retv, ots = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        inv = np.bitwise_not(ots)

        y=930
        x=270
        h=2048
        w=2048

        cropped = inv[y:y+h, x:x+w]

        resized = cv2.resize(cropped,(1024,1024))

        horizontal = np.split(resized, 4, axis=1)

        for idx, h in enumerate(horizontal, start=1):

            vertical = np.split(h, 4, axis=0)

            for ind, v in enumerate(vertical, start=1):

                mean = v.mean()

                if mean >= 8:

                    cv2.imwrite(out_path + 'cvl-' + j[:-4] + '-' + str(idx) + str(ind) + '.png', v)

                else:

                    continue
                    
    print(mode + ' preprocessing done: 100%')

TrainPreprocessing = Preprocessing(mode='train',
                                   in_path='/raw/train/images/input/directory/',
                                   out_path='/preprocessed/train/images/save/directory/')

TestPreprocessing = Preprocessing(mode='test',
                                  in_path='/raw/test/images/input/directory/',
                                  out_path='/preprocessed/test/images/save/directory/')

print('Preprocessing done.')
