import glob
import cv2
import random
import os

pngs = glob.glob('*.png')

bgrs = []

k = number_of_positive_instances_created_in_the_second_step

while len(bgrs) < k:

    j = random.choice(pngs)
    i = random.choice(pngs)

    if j[:4] != i[:4]:

        imgB = cv2.imread(j, 0)
        imgG = cv2.imread(i, 0)
        imgR = cv2.imread('/path/to/the/empty_image.png', 0) 

        imgBGR = cv2.merge((imgB, imgG, imgR))

        cv2.imwrite('/path/to/the/folder/' + j[:-4] + '-' + i, imgBGR)

        bgrs = os.listdir('/path/to/the/folder/')

        print('%.2f%%'%(100*len(bgrs)/k), ' | ', len(bgrs),'/',k, end="\r")


    else:

        continue

else:
    
    print('Done:  100%  | ', len(bgrs), '/', k)
