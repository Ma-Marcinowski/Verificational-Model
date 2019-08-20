import glob
import cv2
from tqdm import tqdm

tifs = glob.glob('*.tif')

for j in tqdm(tifs):

    img = cv2.imread(j, 0)

    y=890
    x=323
    h=2280
    w=2280

    crop = img [y:y+h, x:x+w]

    resized = cv2.resize(crop,(1000,1000))

    cv2.imwrite('/path/to/the/folder/' + j[:-4] + '.png', resized)
