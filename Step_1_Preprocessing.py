import glob
import cv2
import numpy as np
from tqdm import tqdm

tifs = glob.glob('*.tif')

for j in tqdm(tifs, desc='j-loop'):

    img = cv2.imread(j, 0)

    y=890
    x=323
    h=2280
    w=2280

    cropped = img[y:y+h, x:x+w]

    resized = cv2.resize(cropped,(1024,1024))

    horizontal = np.split(resized, 4, axis=1)

    for idx, h in enumerate(horizontal, start=1):

        vertical = np.split(h, 4, axis=0)

        for idn, v in enumerate(vertical, start=1):

            mean = v.mean()

            if mean < 253:

                cv2.imwrite('/path/' + j[:-4] + '-' + str(idx) + str(idn) + '.png', v)

            else:

                continue
