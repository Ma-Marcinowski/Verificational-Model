import os 
import cv2
import glob
import numpy as np
from tqdm import tqdm

os.chdir('/raw/images/directory/')

tifs = glob.glob('*.tif')

for j in tqdm(tifs):

    img = cv2.imread(j, 0) 
    inv = np.bitwise_not(img) 

    y=890
    x=323
    h=2280
    w=2280

    cropped = inv[y:y+h, x:x+w]

    resized = cv2.resize(cropped,(1024,1024))

    horizontal_split = np.split(resized, 4, axis=1)

    for idx, h in enumerate(horizontal_split, start=1):
        
        vertical_split = np.split(h, 4, axis=0)

        for ind, v in enumerate(vertical_split, start=1):

            mean = v.mean()

            if mean >= 4: 

                cv2.imwrite('/preprocessed/images/directory/' + j[:-4] + '-' + str(idx) + str(ind) + '.png', v)

            else:

                continue
