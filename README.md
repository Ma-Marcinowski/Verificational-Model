## Verificational Model / Data Preprocessing

### 0.0. Wprowadzenie
  
   * #### 0.1. Cel, założenia, dane i preprocessing
   
       * 0.1.1. Celem autora niniejszego repozytorium było opracowanie metody preprocesowania obrazów pisma, na potrzeby sztucznych sieci neuronowych do weryfikacji autorstwa dokumentu (poprzez klasyfikację dwóch obrazów pisma do klasy pozytywnej `ten sam autor` albo negatywnej `różni autorzy`).
       
       * 0.1.2. Stąd autor założył iż:
        
            - Obrazy pisma powinny być konkatenowane i wprowadzane do sieci jednym wejściem albo wprowadzane symultanicznie na dwa odrębne wejścia danej sieci.      
            - Obrazy powinny być przetworzone bez znaczącej utraty jakości dla zachowania możliwie największej liczby cech grafizmów.
            
       * 0.1.3. Dane:
       
            - Wykorzystane tutaj zostały obrazy pisma z bazy CVL (F. Kleber, S. Fiel, M. Diem, R. Sablatnig, *CVL-Database: An Off-line Database for Writer Retrieval, Writer Identification and Word Spotting*, "In Proc. of the 12th Int. Conference on Document Analysis and Recognition (ICDAR) 2013" 2013, s. 560 - 564.)
            
       * 0.1.4. Preprocesowanie obrazów polegało na:
               
            - krok pierwszy - przekształcenie obrazów do skali szarości, ekstrakcja przestrzeni pisarskiej z obrazów całych dokumentów, przeskalowanie ekstraktów do wymiarów [1000x1000] pikseli, konwersja obrazów z formatu `tif` na `png`;
            - krok drugi - kombinatoryczna konkatenacja obrazów po dwa (osobno na zbiorze 98 obrazów testowych, osobno na zbiorze 189 obrazów treningowych) w jeden plik graficzny typu RGB (obrazy umieszczane były na odmiennych kanałach przestrzeni barw RGB - jeden na kanale skali czerwieni, drugi na kanale skali zieleni, zaś na kanale skali niebieskiego umieszczany był pusty biały obraz);
            - krok trzeci - konkatenowane obrazy rozdzielane były na podzbiór instancji negatywnych i pozytywnych, dla zbiorów testowego i treningowego (Treningowe: 1134 instancje pozytywne, 34398 instancji negatywnych; Testowe: 194 pozytywne, 9312 negatywnych);
            - krok czwarty (opcjonalny) - ponieważ w drugim kroku wykonywane były wszelkie możliwe kombinacje obrazów dla danego zbioru (testowego / treningowego) to liczba instancji negatywnych znacznie przewyższyła instancje pozytywne, stąd autor dokonał losowego wydzielenia (samplingu) 1134 negatywnych instancji treningowych i 194 negatywnych instancji testowych. 
            - W każdym wypadku krok drugi wykonać można dla kasy negatywnej (pomijając trzeci), albo krok trzeci wykonać dla klasy pozytywnej (pomijając drugi).                  
  
   * #### 0.2. Zastosowane programy
   		
       * 0.2.1. Programy umieszczone w niniejszym repozytorium napisane zostały w języku Python 3.7.3.
  
       * 0.2.2. Wykorzystano biblioteki: OpenCV 4.1.0; tqdm 4.33.0; oraz standardowe biblioteki języka.
  
       * 0.2.3. Aby wykorzystać programy, które autor opracował do preprocesowania obrazów pisma:
  
           - Należy zainstalować Python oraz wskazane biblioteki (metoda instalacji zależy od systemu operacyjnego użytkownika);
           - Należy pobrać programy z niniejszego repozytorium;
           - Następnie otworzyć - za pomocą danego edytora tekstu - pliki zawierające poszczególne programy i dokonać edycji oznaczonych ścieżek, wskazując foldery gdzie kopiowane/zapisywane mają być obrazy pisma (pliki zapisać należy w formacie `py`);
           - Plik z danym programem umieścić należy w folderze, w którym znajdują się obrazy pisma, jakie mają zostać przez dany program przetworzone (jest to najprostsza metoda);
           - Folder, który zawiera program i obrazy, otworzyć należy za pomocą terminala (metoda zależna od systemu operacyjnego);
           - Następnie wpisać należy w terminalu komendę `python3 nazwa_programu.py`, stąd wykonany zostanie wskazany program.
   
   * #### 0.3. Oznaczenie danych
   
       * 0.3.1. Po ukończeniu preprocesowania obrazów pisma, utworzyć należy ich listę w formacie `csv`, która zawierać będzie (na przykładzie AutoML Vis):
       
           - w pierwszej kolumnie oznaczenie przeznaczenia każdego obrazu (trening / test / kroswalidacja), wprowadzone wielkimi literami (`TRAIN` / `TEST` / `VALIDATION`),
           - w drugiej kolumnie ścieżkę obrazu umieszczonego przez użytkownika na Google Cloud Platform (*e.g.* `gs://google_storage_bucket_name/folder_name/0001-1-0001-2.png`),
           - w trzeciej kolumnie klasę (*label*) obrazu (*e.g.* `Pozytywna` / `Negatywna`).
           
       * 0.3.2. Plik typu `csv` utworzyć można w dowolnym programie typu *spreadsheet* (*e.g.* *calc* / *excel*).
       
       * 0.3.3. Metoda symultanicznego uzyskania wszystkich ścieżek - z nazwami - obrazów zależna jest od systemu operacyjnego użytkownika. 
       
            - Odnotować warto, że w przypadku, gdy obrazy różnych klas znajdują się w różnych folderach to oznaczenie ich klas jest niezwykle proste (odbywa się na podstawie ich ścieżki). To samo dotyczy przeznaczenia do treningu / testu / kroswalidacji.
            - Przypomnieć należy, iż w przypadku umieszczenia obrazów na chmurze jest konieczne, aby w pliku `csv` edytować ścieżki do folderów (*e.g.* `C:\database_folder\folder_name\0001-1-0001-2.png` za pomocą `find and replace` zamienić na `gs://google_storage_bucket_name/folder_name/0001-1-0001-2.png`).         
  
### 1.0. Krok pierwszy
### 2.0. Krok drugi
### 3.0. Krok trzeci

