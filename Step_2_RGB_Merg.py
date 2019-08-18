import glob
import cv2
from tqdm import tqdm

#Format obraz pisma został zmieniony z {.tif} na {.png} przez wykonanie {Extr_Res_Gray.py}.
pngs = glob.glob('*.png')

#Dla j w zakresie pngs, gdzie {tqdm(zakres, desc='opis')} wyświetlać będzie status realizacji j-tej pętli.
for j in tqdm(pngs, desc='j-loop'):
    #Dla i w zakresie pngs, jest i-tą pętlą umieszczoną w j-tej pętli (tzw. nested loop), gdzie {leave=False} usuwa pasek statusu wykonanej już i-tej pętli.
    #Zakładając że w zakresie pngs znajduje się 10 obrazów w formacie {.png}, to i-ta pętla zostanie wykonana ogółem 100 razy (10 razy dla każdego {j}),
    #zaś j-ta pętla wykonana zostanie 10 razy.
    for i in tqdm(pngs, desc='i-loop', leave=False):

            #Jeżeli argument {j} jest tożsamy z {i}.
            if j == i:

                #Wyjdź z danej i-tej pętli i przejdź do następnej kombinacji argumentów.
                continue

            #Jeżeli poprzedni warunek nie został spełniony, to różne od siebie obrazy pisma {i} oraz {j} zostaną wczytane.
            imgR = cv2.imread(j, 0)
            imgG = cv2.imread(i, 0)
            #Oraz wczytany zostanie jeden pusty/biały obraz.
            imgB = cv2.imread('/path/Empty.png', 0)

            #Obrazy te zostaną następnie scalone w jeden plik graficzny, ale na odrębnych kanałach przestrzeni barw RGB - jeżeli ich wymiary są takie same.
            #Odpowiednio w skali czerwieni (red), skali zieleni (green), skali niebieskiego (blue) - w ostatnim przypadku jest to pusty obraz.
            imgRGB = cv2.merge((imgR, imgG, imgB))

            #Tak skombinowane obrazy pisma (umieszczone na odrębnych kanałach jednego pliku) zapisane zostaną z zachowaniem ich nazw i formatu,
            #poprzez scalenie nazwy {j} bez określenia formatu, z nazwą {i} które zawiera określenie formatu, oraz znakiem {-} pomiędzy.
            cv2.imwrite('/path/' + j[:-4] + '-' + i, imgRGB)
