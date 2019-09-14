import glob
import os
import csv
import random
import sklearn
import pandas as pd
from tqdm import tqdm

pngs = glob.glob('*.png')

with open('/path/Dataframe.csv', 'a+') as f:

    writer = csv.writer(f)

    for j in tqdm(pngs, leave=False):
        for i in pngs:

            if j[:4] == i[:4] and j != i:

                pair = ['/path/' + j, '/path/' + i, 1]

                writer.writerow(pair)

            else:

                continue
    
    print('Done positive: 100%')
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

        print('Done negative: 100%')

df = pd.read_csv('/path/Dataframe.csv', header=None, names = ['Leftname', 'Rightname', 'Label'])

df = sklearn.utils.shuffle(df)

df.columns = ["Leftname", "Rightname", "Label"]

df.to_csv('/path/Dataframe.csv', index=False)

print('Dataframe done.') 
