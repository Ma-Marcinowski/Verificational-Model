#Polecenie {import} pozwala wykorzystać określony moduł/funkcję poprzez importowanie go z dostępnej biblioteki.
import glob
import cv2
from tqdm import tqdm

#Funkcja {glob.glob('dowolna_nazwa.format_pliku')} dokonuje identyfikacji wszystkich plików o danym formacie w folderze gdzie wykonywany jest program.
tifs = glob.glob('*.tif')

#Dla danego j w zakresie tifs,
#gdzie funkcja {tqdm(zakres)} wyświetlać będzie procentowy status realizacji pętli {for x in y:} ze względu na wszystkie j w tifs.
for j in tqdm(tifs):

    #Wczytaj j w skali szarości,
    #innymi słowy {cv2.imread(wczytywany obraz, zakres barw)}.
    img = cv2.imread(j, 0)

    #Współrzędne początkowe w pikselach, gdzie {y} określa wysokość, a {x} szerokość - od lewego górnego rogu obrazu.
    y=890
    x=323
    #Współrzędne końcowe w pikselach, gdzie {h} określa wysokość, a {w} szerokość - od lewego górnego rogu obrazu.
    h=2280
    w=2280

    #Funkcja {obraz_wczytany_w_skali_szarości [y:y+h, x:x+w]} wycina fragment obrazu określony współrzędnymi {od y do h, od x do w}.
    crop = img [y:y+h, x:x+w]

    #Funckja {cv2.resize(wycięty_obraz, (pożądana_szerokość, pożodana_wysokość)} skaluje obraz do wymiarów wskazanych w pikselach.
    resized = cv2.resize(crop,(1000,1000))

    #Funkcja {cv2.imwrite('ścieżka/do/folderu' + nazwa_j-tego_elementu[minus cztery ostatnie znaki] + 'wybrane_znaki.wybrany_format', przeskalowany_obraz)}
    #pozwala zapisać wybrany obraz (tutaj wczytany w skali szarości, następnie przycięty i przeskalowany) do wybranego folderu,
    #pod swoją nazwą,w wybranym formacie.
    cv2.imwrite('/path/to/the/folder/' + j[:-4] + '.png', resized)
