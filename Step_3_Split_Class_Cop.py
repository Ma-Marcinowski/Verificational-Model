import glob
import shutil
from tqdm import tqdm

pngs = glob.glob('*.png')

for j in tqdm(pngs):

	#Ponieważ identyfikator autora danego dokumentu określają cztery pierwsze cyfry nazwy obrazu (w przypadku CVL).
	#Jeżeli identyfikatory określające autorów (dokumentów, które skombinowano w jeden plik graficzny) są tożsame, to:
	if j[:4] == j[7:-6]:

		#Funkcja {shutil.copy2(kopiowany_plik, 'kopiowany/do/folderu/')}
		#skopiuje obraz {j} do folderu utworzonego dla klasy pozytywnej,
		shutil.copy2(j, '/path/training_data/positive/')

	#a w odmiennym przypadku,
	else:

		#skopiuje obraz {j} do folderu utworzonego dla klasy pozytywnej.
		shutil.copy2(j, '/path/training_data/negative/')
