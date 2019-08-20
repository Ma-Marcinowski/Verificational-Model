import glob
import cv2
import random
from tqdm import tqdm

pngs = glob.glob('*.png')

k = 0

pbar = tqdm(total=number_of_negative_instances)

while k < number_of_negative_instances:

    j = random.choice(pngs)
    i = random.choice(pngs)

    if j[:4] != i[:4]:

        imgR = cv2.imread(j, 0)
        imgG = cv2.imread(i, 0)
        imgB = cv2.imread('/path/to/the/empty_image.png', 0) 

        imgRGB = cv2.merge((imgR, imgG, imgB))

        cv2.imwrite('/path/to/the/folder/' + j[:-4] + '-' + i, imgRGB)

        k += 1

        pbar.update(1)

    else:

        continue
      
pbar.close()
