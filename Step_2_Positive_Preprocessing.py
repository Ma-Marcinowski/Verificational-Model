import glob
import cv2
from tqdm import tqdm

pngs = glob.glob('*.png')

for j in tqdm(pngs, desc='j-loop'):
    for i in tqdm(pngs, desc='i-loop', leave=False):

            if j[:4] == i[:4] and j != i:

            	imgB = cv2.imread(j, 0)
            	imgG = cv2.imread(i, 0)
            	imgR = cv2.imread('/path/to/the/empty_image.png', 0) 

            	imgBGR = cv2.merge((imgB, imgG, imgR))

            	cv2.imwrite('/path/to/the/folder/' + j[:-4] + '-' + i, imgBGR)

            else:

            	continue   
