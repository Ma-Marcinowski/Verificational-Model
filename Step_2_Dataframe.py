import glob
import os
import csv
import random
from tqdm import tqdm

pngs = glob.glob('*.png')

with open('/path/Dataframe.csv', 'a+') as f:

    writer = csv.writer(f)

    for j in tqdm(pngs):
        for i in pngs:

            if j[:4] == i[:4] and j != i:

                pair = ['/path/' + j, '/path/' + i, 1]

                writer.writerow(pair)

            else:

                continue

    g = f.tell()
    k = 2 * f.tell()

    while g < k:

        j = random.choice(pngs)
        i = random.choice(pngs)

        if j[:4] != i[:4]:

            pair = ['/path/' + j, '/path/' + i, 0]

            writer.writerow(pair)

            g = f.tell()
            
            print('%.2f%%'%(100*g/k), end="\r")

        else:

            continue

    else:

        print('Done 100%')
