## Verificational Model

### 1. Preprocessing
  
   * #### 1.1. Cel, założenia, dane i preprocessing
   
       * 1.1.1. Celem autora niniejszego repozytorium było opracowanie metody preprocesowania obrazów pisma, na potrzeby sztucznych sieci neuronowych do weryfikacji autorstwa dokumentu (poprzez klasyfikację dwóch obrazów pisma do klasy pozytywnej `ten sam autor` albo negatywnej `różni autorzy`).
       
       * 1.1.2. Stąd autor założył iż:
        
            - Obrazy pisma powinny być wprowadzane symultanicznie na dwa odrębne wejścia danej sieci.      
            - Obrazy powinny być przetworzone bez znaczącej utraty jakości dla zachowania możliwie największej liczby cech grafizmów.
            
       * 1.1.3. Dane:
       
            - Wykorzystane tutaj zostały obrazy pisma z bazy CVL (F. Kleber, S. Fiel, M. Diem, R. Sablatnig, *CVL-Database: An Off-line Database for Writer Retrieval, Writer Identification and Word Spotting*, "In Proc. of the 12th Int. Conference on Document Analysis and Recognition (ICDAR) 2013" 2013, s. 560 - 564).
            
       * 1.1.4. Preprocesowanie obrazów polegało na:
               
            - Krok pierwszy - przekształcenie obrazów (skany całych dokumentów) do skali szarości, ekstrakcja przestrzeni pisarskiej z obrazów, przeskalowanie ekstraktów do wymiarów [1000x1000] pikseli, konwersja obrazów z formatu `tif` na `png`;
            - Krok drugi - **kombinatoryczna konkatenacja** obrazów po dwa (osobno na zbiorze obrazów testowych, osobno na zbiorze obrazów treningowych) w jeden plik graficzny typu BGR (obrazy umieszczane były na odmiennych kanałach przestrzeni barw BGR - jeden na kanale skali niebieskiego, drugi na kanale skali zieleni, zaś na kanale skali czerwieni umieszczany był pusty biały obraz). Pary tworzone w kroku drugim należą do klasy pozytywnej (`ten sam autor`), ponieważ warunkiem konkatenacji jest zbieżność identyfikatorów (cztery pierwsze cyfry nazwy obrazu oznaczające jego autora);
            - Krok trzeci - ponieważ liczba możliwych kombinacji w pary jest znacznie większa dla klasy negatywnej, niż dla klasy pozytywnej (*e.g.* na zbiorze testowym bazy CVL - 189 obrazów - będą to 1134 instancje pozytywne i 34398 instancji negatywnych), stąd łączenie obrazów w pary należące do klasy negatywnej (`różni autorzy`) odbywa się jako proces **losowej konkatenacji** obrazów w pary negatywne (warunkiem utworzenia pary jest rozbieżność identyfikatorów), aż do utworzenia danej `k` liczby par, którą określić należy w treści programu (najbardziej pożądana jest liczba `k` równa liczbie utworzonych dotąd instancji pozytywnych);
            - W każdym wypadku krok drugi wykonać można dla kasy negatywnej (zamiast trzeciego), albo krok trzeci wykonać dla klasy pozytywnej (zamiast drugiego).                   
  
   * #### 1.2. Zastosowane programy
   		
       * 1.2.1. Programy umieszczone w niniejszym repozytorium napisane zostały w języku Python 3.7.3.
  
       * 1.2.2. Wykorzystano biblioteki: OpenCV 4.1.0; tqdm 4.33.0; oraz standardowe biblioteki języka.
  
       * 1.2.3. Aby wykorzystać programy, które autor opracował do preprocesowania obrazów pisma:
  
           - Należy zainstalować Python oraz wskazane biblioteki (metoda instalacji zależy od systemu operacyjnego użytkownika);
           - Należy pobrać programy z niniejszego repozytorium;
           - Następnie otworzyć - za pomocą danego edytora tekstu - pliki zawierające poszczególne programy i dokonać edycji oznaczonych ścieżek, wskazując foldery gdzie kopiowane/zapisywane mają być obrazy pisma (pliki zapisać należy w formacie `py`);
           - Plik z danym programem umieścić należy w folderze, w którym znajdują się obrazy pisma, jakie mają zostać przez dany program przetworzone (jest to najprostsza metoda);
           - Folder, który zawiera program i obrazy, otworzyć należy za pomocą terminala / interpretera poleceń (metoda zależna od systemu operacyjnego);
           - Następnie wpisać należy w terminalu komendę `python3 nazwa_programu.py`, stąd wykonany zostanie wskazany program;
           - Gdyby zaistniała taka konieczność: aby przerwać wykonywanie programu wykorzystać można w terminalu kombinację klawiszy `Ctrl + C`, aby zawiesić wykonywanie programu `Ctrl + Z`, aby zaś przywrócić wykonywanie zawieszonego programu wpisać należy komendę `fg` (dla terminalów większości systemów operacyjnych, *e.g.* macOS, Linux).  
   
   * #### 1.3. Oznaczenie danych
   
       * 1.3.1. Po ukończeniu preprocesowania obrazów pisma, utworzyć należy ich listę w formacie `csv`, która zawierać będzie (na przykładzie AutoML Vision Beta):
       
           - w pierwszej kolumnie oznaczenie przeznaczenia każdego obrazu (trening / test / kroswalidacja), wprowadzone wielkimi literami (`TRAIN` / `TEST` / `VALIDATION`),
           - w drugiej kolumnie ścieżkę obrazu umieszczonego przez użytkownika na Google Cloud Platform (*e.g.* `gs://google_storage_bucket_name/folder_name/0001-1-0001-2.png`),
           - w trzeciej kolumnie klasę (*label*) obrazu (*e.g.* `Pozytywna` / `Negatywna`).
           
       * 1.3.2. Plik typu `csv` utworzyć można w dowolnym programie typu *spreadsheet* (*e.g.* *calc* / *excel*).
       
       * 1.3.3. Metoda symultanicznego uzyskania wszystkich ścieżek - z nazwami - obrazów zależna jest od systemu operacyjnego użytkownika. 
       
            - Odnotować warto, że w przypadku, gdy obrazy różnych klas znajdują się w różnych folderach to oznaczenie ich klas jest niezwykle proste (odbywa się na podstawie ich ścieżki). To samo dotyczy przeznaczenia do treningu / testu / kroswalidacji.
            - Przypomnieć należy, iż w przypadku umieszczenia obrazów na chmurze jest konieczne, aby w pliku `csv` edytować ścieżki do folderów (*e.g.* `C:\database_folder\folder_name\0001-1-0001-2.png` za pomocą `find and replace` zamienić na `gs://google_storage_bucket_name/folder_name/0001-1-0001-2.png`).         
  
### 2. Model Weryfikacyjny
### 3. Ewaluacja Modelu

