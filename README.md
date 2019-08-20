## Verificational Model / Data Preprocessing

### 1.0. Wprowadzenie
  
   * #### 1.1. Cel i założenia
  
   * #### 1.2. Zastosowane programy
   		
       * 1.2.1. Programy umieszczone w niniejszym repozytorium napisane zostały w języku Python 3.7.3.
  
       * 1.2.2. Wykorzystano biblioteki: OpenCV 4.1.0; tqdm 4.33.0; oraz standardowe biblioteki języka.
  
       * 1.2.3. Aby wykorzystać programy, które autor opracował do preprocesowania obrazów pisma:
  
           - Należy zainstalować Python oraz wskazane biblioteki (metoda instalacji zależy od systemu operacyjnego użytkownika);
           - Należy pobrać programy z niniejszego repozytorium;
           - Następnie otworzyć - za pomocą danego edytora tekstu - pliki zawierające poszczególne programy i dokonać edycji oznaczonych ścieżek, wskazując foldery gdzie kopiowane/zapisywane mają być obrazy pisma (pliki zapisać należy w formacie `py`);
           - Plik z danym programem umieścić należy w folderze, w którym znajdują się obrazy pisma, jakie mają zostać przez dany program przetworzone (jest to najprostsza metoda);
           - Folder, który zawiera program i obrazy, otworzyć należy za pomocą terminala (metoda zależna od systemu operacyjnego);
           - Następnie wpisać należy w terminalu komendę `python3 nazwa_programu.py`, stąd wykonany zostanie wskazany program.
   
   * #### 1.3. Oznaczenie danych
   
       * 1.3.1. Po ukończeniu preprocesowania obrazów pisma, utworzyć należy ich listę w formacie `csv`, która zawierać będzie (na przykładzie AutoML Vis):
           - w pierwszej kolumnie oznaczenie przeznaczenia każdego obrazu (trening / test / kroswalidacja), wprowadzone wielkimi literami (`TRAIN` / `TEST` / `VALIDATION`),
           - w drugiej kolumnie ścieżkę obrazu umieszczonego przez użytkownika na Google Cloud Platform (*e.g.* `gs://google_storage_bucket_name/folder_name/0001-1-0001-2.png`),
           - w trzeciej kolumnie klasę (*label*) obrazu (*e.g.* `Pozytywna` / `Negatywna`).
       * 1.3.2. Plik typu `csv` utworzyć można w dowolnym programie typu *spreadsheet* (*e.g.* *calc* / *excel*).
       * 1.3.3. Metoda symultanicznego uzyskania wszystkich ścieżek - z nazwami - obrazów zależna jest od systemu operacyjnego użytkownika. 
            - Odnotować warto, że w przypadku, gdy obrazy różnych klas znajdują się w różnych folderach to oznaczenie ich klas jest niezwykle proste (odbywa się na podstawie ich ścieżki). To samo dotyczy przeznaczenia do treningu / testu / kroswalidacji.
            - Przypomnieć należy, iż w przypadku umieszczenia obrazów na chmurze jest konieczne, aby w pliku `csv` edytować ścieżki do folderów (*e.g.* `C:\database_folder\folder_name\0001-1-0001-2.png` za pomocą `find and replace` zamienić na `gs://google_storage_bucket_name/folder_name/0001-1-0001-2.png`).         
  
### 2.0. Krok pierwszy
### 3.0. Krok drugi
### 4.0. Krok trzeci
### 5.0. Krok czwarty (opcjonalny)
