## Verificational Model

### 1. Preprocessing
  
   * #### 1.1. Cel, założenia, dane i kroki preprocessingu
   
       * 1.1.1. Celem autora niniejszego repozytorium było opracowanie metody preprocesowania obrazów pisma, na potrzeby sztucznych sieci neuronowych do weryfikacji autorstwa dokumentu (poprzez klasyfikację dwóch obrazów pisma do klasy pozytywnej `ten sam autor` albo negatywnej `różni autorzy`).
       
       * 1.1.2. Stąd autor założył iż:
        
            - Obrazy pisma powinny być wprowadzane parami / symultanicznie na dwa odrębne wejścia danej sieci.      
            - Obrazy powinny być przetworzone bez znaczącej utraty jakości dla zachowania możliwie największej liczby cech grafizmów.
            
       * 1.1.3. Dane:
       
            - Wykorzystane tutaj zostały obrazy pisma (skany całych dokumentów) z bazy CVL (F. Kleber, S. Fiel, M. Diem, R. Sablatnig, *CVL-Database: An Off-line Database for Writer Retrieval, Writer Identification and Word Spotting*, "In Proc. of the 12th Int. Conference on Document Analysis and Recognition (ICDAR) 2013" 2013, s. 560 - 564). Gdzie zbiór treningowy zawiera 1415 obrazów pisma, testowy zaś 189.
            
       * 1.1.4. Preprocesowanie obrazów polegało na:
               
            - Krok pierwszy - `Step_1_Preprocessing.py` - przekształcenie obrazów (skany całych dokumentów) do skali szarości, ekstrakcja przestrzeni pisarskiej z obrazów, przeskalowanie ekstraktów do wymiarów [1024x1024] piksele, dalszy podział ekstraktów na komórki o wymiarach [256x256] pikseli, konwersja z formatu `tif` na `png`. Ostateczne ekstrakty, które nie zawierają lub zawierają nieznaczą ilość tekstu, pomijane są przez program - w każdym razie posortować można utworzone pliki według ich rozmiaru;
            - Krok drugi - `Step_2_Dataframe.py` - utworzenie *dataframe* (plik typu `csv`, który edytować można w dowolnym programie typu *spreadsheet* (*e.g.* *calc* / *excel*).) rozłącznie dla zbioru testowego i treningowego, poprzez kombinatoryczne zestawienie nazw obrazów pisma w pary, które należą do klasy pozytywnej, oraz losowe zestawienie nazw obrazów pisma w pary, które należą do klasy negatywnej (liczba możliwych kombinacji negatywnych jest znacznie większa niż pozytywnych, zatem tworzone są najpierw wszystkie instancje pozytywne, a następnie losowane są instancje negatywne, których liczba jest równa liczbie instancji pozytywnych). Zestawienie nazw obrazów odbywa się wierszami, według kolumn `lewy kanał sieci, prawy kanał sieci, klasa` (o przynależności do klasy `ten sam autor` lub `różni autorzy` przesądza zbieżność lub rozbieżność identyfikatorów - cztery pierwsze cyfry nazwy obrazu oznaczające jego autora). Metoda wymaga aby: obrazy testowe znajdowały się w odmiennym folderze niż obrazy treningowe, a do nazw obrazów dołączyć (aktualne lub zamieżone) ścieżki ich folderów (wystarczy wskazać je w treści programu, a w każdym razie edytować je można za pomocą `find and replace` w dowolnym programie typu *spreadsheet*). Nie jest natomiast konieczne aby utworzyć uprzednio plik `csv`, gdyby bowiem nie istniał, zostanie utworzony przez program (gdyby zaś taki plik uprzednio utworzyć, naley wskazać w treści programu jego ścieżkę i nazwę);
            - Krok trzeci - ponieważ opracowany model polega na sekwencyjnym wprowadzeniu do ANN obrazów parami (*de facto* wierszami), to wczytywanie ich powinno odbywać się sekwencyjnie, *ergo* według kolejności określonej wierszami *dataframe*, dlatego też przetasowanie par i wylosowanie zbioru walidacyjnego przeprowadzić należy w ramach *dataframe*. Plik `csv` otworzyć należy danym programem typu *spreadsheet*, następnie wypełnić należy komórki dodatkowej (czwartej) kolumny formułą `=Rand()` (która wygeneruje w komórkach dodatkowej kolumny liczby pseudolosowe z przedziału [0, 1]) i zaznaczyć wszystkie cztery kolumny, a następnie przesortować je wierszami. Sortowanie (rosnące lub malejące) zapewni losowe przetasowanie trzech pierwszych kolumn wierszami, ze względu na pseudolosowe liczby w czwartej kolumnie (którą można potem usunąć). Ostatecznie dodać należy nagółwki kolumn (*e.g.* `lewy kanał sieci, prawy kanał sieci, klasa`). Krok trzeci przeprowadzić należy zarówno dla *dataframe* zbioru treningowego i testowego;
            - Krok czwarty - po przeprowadzeniu kroku trzeciego, przystąpić można do utworzenia *dataframe* zbioru walidacyjnego. Otóż, utworzyć należy nowy plik typu `csv`, a następnie przenieść do niego dowolną liczbę wierszy z *dataframe* zbioru **treningowego** (zazwyczaj 10% do 20% całkowitej liczby instancji treningowych) chociażby daną liczbę wierszy od końca, gdyż zostały już losowo przetasowane. Należy jednak zwrócić uwagę, aby liczba instancji pozytywnych zblilżona była do liczby instancji negatywnych. Nie należy tworzyć następnie osobnego folderu obrazów walidacyjnych, ani nie jest konieczne, aby zachować dalej odrębne foldery dla obrazów testowych i treningowych - jeżeli preprocesowanie przeprowadzone zostało według lub analogicznie do kroków poprzednich - bowiem przynależność danej pary obrazów do danego zbioru określa na obecnym etapie *dataframe* danego zbioru. 
  
   * #### 1.2. Zastosowane programy
   		
       * 1.2.1. Programy umieszczone w niniejszym repozytorium napisane zostały w języku Python 3.7.3.
  
       * 1.2.2. Wykorzystano biblioteki: OpenCV 4.1.0; Numpy 1.17.0; tqdm 4.33.0; oraz standardowe biblioteki języka.
  
       * 1.2.3. Aby wykorzystać programy, które autor opracował do preprocesowania obrazów pisma:
  
           - Należy zainstalować Python oraz wskazane biblioteki (metoda instalacji zależy od systemu operacyjnego użytkownika);
           - Należy pobrać programy z niniejszego repozytorium;
           - Następnie otworzyć - za pomocą danego edytora tekstu - pliki zawierające poszczególne programy i dokonać edycji oznaczonych ścieżek, wskazując foldery gdzie zapisywane mają być obrazy pisma / dataframe (pliki programów zapisać należy w formacie `py`);
           - Plik z danym programem umieścić należy w folderze, w którym znajdują się obrazy pisma, jakie mają zostać przez dany program przetworzone (jest to najprostsza metoda);
           - Folder, który zawiera program i obrazy, otworzyć należy za pomocą terminala / interpretera poleceń (metoda zależna od systemu operacyjnego);
           - Następnie wpisać należy w terminalu komendę `python3 nazwa_programu.py`, stąd wykonany zostanie wskazany program;
           - Gdyby zaistniała taka konieczność: aby przerwać wykonywanie programu wykorzystać można w terminalu kombinację klawiszy `Ctrl + C`, aby zawiesić wykonywanie programu `Ctrl + Z`, aby zaś przywrócić wykonywanie zawieszonego programu wpisać należy komendę `fg` (dla terminalów większości systemów operacyjnych, *e.g.* macOS, Linux).      
  
### 2. Model Weryfikacyjny
### 3. Ewaluacja Modelu

