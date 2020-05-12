import os
import cv2
import numpy as np
from tqdm import tqdm

def Preprocessing(mode, in_path, out_path_256, out_path_512, filters_dir_256, filters_dir_512):

    k_names_256 = os.listdir(filters_dir_256)
    kernels_256 = [filters_dir_256 + n for n in k_names_256]
    
    k_names_512 = os.listdir(filters_dir_512)
    kernels_512 = [filters_dir_512 + n for n in k_names_512]

    img_names = os.listdir(in_path)
    img_paths = [in_path + n for n in img_names]
    listed = zip(img_paths, img_names)

    for j, i in tqdm(listed, total=len(img_paths), desc=mode+'-loop'):

        img = cv2.imread(j, 0)

        inv = np.bitwise_not(img)

        y=930
        x=270
        h=2048
        w=2048

        cropped = inv[y:y+h, x:x+w]

        resized = cv2.resize(cropped,(1024,1024))

        horizontal_256 = np.split(resized, 4, axis=1)

        for idx, h in enumerate(horizontal_256, start=1):

            vertical_256 = np.split(h, 4, axis=0)

            for ind, v in enumerate(vertical_256, start=1):

                thv, denv = cv2.threshold(v, 55, 255, cv2.THRESH_TOZERO)

                filtered = []

                for kern in kernels_256:

                    kernel = cv2.imread(kern, 0)

                    ksum = np.sum(np.multiply(denv, kernel))

                    filtered.append(ksum)

                if 0 not in filtered:

                    cv2.imwrite(out_path_256 + 'cvl-' + i[:-4] + '-' + str(idx) + str(ind) + '.png', v)

                else:

                    continue
                
        horizontal_512 = np.split(resized, 2, axis=1)

        for idx, h in enumerate(horizontal_512, start=1):

            vertical_512 = np.split(h, 2, axis=0)

            for ind, v in enumerate(vertical_512, start=1):

                thv, denv = cv2.threshold(v, 55, 255, cv2.THRESH_TOZERO)

                filtered = []

                for kern in kernels_512:

                    kernel = cv2.imread(kern, 0)

                    ksum = np.sum(np.multiply(denv, kernel))

                    filtered.append(ksum)

                if 0 not in filtered:

                    cv2.imwrite(out_path_512 + 'cvl-' + i[:-4] + '-' + str(idx) + str(ind) + '.png', v)

                else:

                    continue

    print(mode + ' preprocessing done: 100%')

TrainPrep = Preprocessing(mode='train',
                          in_path='/raw/train/images/input/directory/',
                          out_path_256='/preprocessed/256x256/train/images/save/directory/',
                          out_path_512='/preprocessed/c/train/images/save/directory/',                         filters_dir_256='256x256//filters/directory/',
                          filters_dir_512='512x512//filters/directory/')

TestPrep = Preprocessing(mode='test',
                         in_path='/raw/test/images/input/directory/',
                         out_path_256='/preprocessed/256x256/test/images/save/directory/',
                         out_path_512='/preprocessed/512x512/test/images/save/directory/',   
                         filters_dir_256='256x256//filters/directory/',
                         filters_dir_512='512x512//filters/directory/')

print('Preprocessing done.')

