import glob
import shutil
import random
from tqdm import tqdm

pngs = glob.glob('*.png')

#Funkcję {random.sample(zakres, k=liczba_losowanych_elementów_ze_wskazanego_zakresu),
#autor wykorzystał do wylosowania ze zbioru negatywnych instancji, takiej ich liczby,
#która odpowiada liczbie pozytywnych instancji.
neg_samp = tqdm(random.sample(pngs, k=1134), desc='Sampling')

for j in tqdm(neg_samp, desc='Copying'):

	shutil.copy2(j, '/path/sample/of/negative')
