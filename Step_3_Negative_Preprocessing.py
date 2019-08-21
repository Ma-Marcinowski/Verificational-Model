import glob
import cv2
import random
import os

pngs = glob.glob('*.png')

rgbs = []

k = number_of_intended_negative_instances

while len(rgbs) < k:

    j = random.choice(pngs)
    i = random.choice(pngs)

    if j[:4] != i[:4]:

        imgR = cv2.imread(j, 0)
        imgG = cv2.imread(i, 0)
        imgB = cv2.imread('/path/to/the/empty_image.png', 0) 

        imgRGB = cv2.merge((imgR, imgG, imgB))

        cv2.imwrite('/path/to/the/folder/' + j[:-4] + '-' + i, imgRGB)

        rgbs = os.listdir('/path/to/the/folder/')

        print('%.2f%%'%(100*len(rgbs)/k), ' | ', len(rgbs),'/',k, end="\r")


    else:

        continue

else:
    
    print('Done:  100%  | ', len(rgbs), '/', k)
