import glob
import os
import random

pngs = glob.glob('*.png')

with open('/path/Dataframe.csv', 'a+') as f:

    writer = csv.writer(f)

    for j in pngs:
        for i in pngs:

            if j[:4] == i[:4] and j != i:

                pair = ['/path/' + j, '/path/' + i, 'Poz']

                writer.writerow(pair)

            else:

                continue

    g = f.tell()
    k = 2 * f.tell()

    while g < k:

        j = random.choice(pngs)
        i = random.choice(pngs)

        if j[:4] != i[:4]:

            pair = ['/path/' + j, '/path/' + i, 'Neg']

            writer.writerow(pair)

            g = f.tell()

        else:

            continue

    else:

        print('Done.')
