## Verificational Model

### 1. Preprocessing
  
   * #### 1.1. Objective, assumptions, database and steps of preprocessing
   
     * 1.1.1. Objective of repository author's was to prepare a method of image preprocessing for verification of offline handwritten documents authorship by artificial neural network (through classification of preprocessed image pairs to positive `same author` or negative `different authors` class).
       
     * 1.1.2. Author's assumptions were that:
     
        - Document images ought to be processed simultaneously by two separate convolutional neural networks and classified by one multilayer perceptron (fully connected layers);
        
        - Preprocessing shouldn't affect image quality to preserve most of handwriting features.
            
     * 1.1.3. Database:
       
        - Subsets of 1415 training and 189 test documents (full page scans) from CVL database (F. Kleber, S. Fiel, M. Diem, R. Sablatnig, *CVL-Database: An Off-line Database for Writer Retrieval, Writer Identification and Word Spotting*, "In Proc. of the 12th Int. Conference on Document Analysis and Recognition (ICDAR) 2013" 2013, p. 560 - 564). 
            
     * 1.1.4. Steps of preprocessing:
               
        - Krok pierwszy - `Step_1_Preprocessing.py` - przekształcenie obrazów (skany całych dokumentów) do skali szarości (skala od czarny=0 do biały=255), inwersja kolorów, ekstrakcja przestrzeni pisarskiej z obrazów, przeskalowanie ekstraktów do wymiarów [1024x1024] piksele, dalszy podział ekstraktów na komórki o wymiarach [256x256] pikseli, konwersja z formatu `tif` na `png`. Ostateczne ekstrakty, które nie zawierają lub zawierają nieznaczą ilość tekstu, pomijane są przez program na podstawie progu średniej wartości pikseli - w każdym razie posortować można utworzone pliki według ich rozmiaru i na tej podstawie usunąć;
            
        - Krok drugi - `Step_2_Dataframe.py` - utworzenie *dataframe* (plik typu `csv`, który edytować można w dowolnym programie typu *spreadsheet* (*e.g.* *calc* / *excel*).) rozłącznie dla zbioru testowego i treningowego, poprzez kombinatoryczne zestawienie nazw obrazów pisma w pary, które należą do klasy pozytywnej, oraz losowe zestawienie nazw obrazów pisma w pary, które należą do klasy negatywnej (liczba możliwych kombinacji negatywnych jest znacznie większa niż pozytywnych, zatem tworzone są najpierw wszystkie instancje pozytywne, a następnie losowane są instancje negatywne, których liczba jest równa liczbie instancji pozytywnych). Zestawienie nazw obrazów odbywa się wierszami, według kolumn `lewy kanał sieci, prawy kanał sieci, klasa` (o przynależności do klasy `ten sam autor` lub `różni autorzy` przesądza zbieżność lub rozbieżność identyfikatorów - cztery pierwsze cyfry nazwy obrazu oznaczające jego autora). Metoda wymaga aby: obrazy testowe znajdowały się w odmiennym folderze niż obrazy treningowe, a do nazw obrazów dołączyć (aktualne lub zamieżone) ścieżki ich folderów (wystarczy wskazać je w treści programu, a w każdym razie edytować je można za pomocą `find and replace` w dowolnym programie typu *spreadsheet*). Nie jest natomiast konieczne aby utworzyć uprzednio plik `csv`, gdyby bowiem nie istniał, zostanie utworzony przez program (gdyby zaś taki plik uprzednio utworzyć, naley wskazać w treści programu jego ścieżkę i nazwę);
            
        - Krok trzeci - ponieważ opracowany model polega na sekwencyjnym wprowadzeniu do ANN obrazów parami (*de facto* wierszami), to wczytywanie ich powinno odbywać się sekwencyjnie, *ergo* według kolejności określonej wierszami *dataframe*, dlatego też przetasowanie par i wylosowanie zbioru walidacyjnego przeprowadzić należy w ramach *dataframe*. Plik `csv` otworzyć należy danym programem typu *spreadsheet*, następnie wypełnić należy komórki dodatkowej (czwartej) kolumny formułą `=Rand()` (która wygeneruje w komórkach dodatkowej kolumny liczby pseudolosowe z przedziału [0, 1]) i zaznaczyć wszystkie cztery kolumny, a następnie przesortować je wierszami - konieczne jest aby kluczem sortowania (*Sort Key*) ustawić czwartą kolumnę. Sortowanie (rosnące lub malejące) zapewni losowe przetasowanie trzech pierwszych kolumn wierszami, ze względu na pseudolosowe liczby w czwartej kolumnie (którą można potem usunąć). Ostatecznie dodać należy nagółwki kolumn (*e.g.* `lewy kanał sieci, prawy kanał sieci, klasa`). Krok trzeci przeprowadzić należy zarówno dla *dataframe* zbioru treningowego i testowego;
            
        - Krok czwarty - po przeprowadzeniu kroku trzeciego, przystąpić można do utworzenia *dataframe* zbioru walidacyjnego (który przeznaczony jest do testowania modelu podczas jego nauki). Otóż, utworzyć należy nowy plik typu `csv`, a następnie skopiować do niego dowolną liczbę wierszy z *dataframe* zbioru **testowego** (zazwyczaj 10% do 20% całkowitej liczby instancji testowych) chociażby daną liczbę wierszy od końca, gdyż zostały już losowo przetasowane. Należy jednak zwrócić uwagę, aby liczba instancji pozytywnych zblilżona była do liczby instancji negatywnych. Nie należy tworzyć następnie osobnego folderu obrazów walidacyjnych, ani nie jest konieczne, aby zachować dalej odrębne foldery dla obrazów testowych i treningowych - jeżeli preprocesowanie przeprowadzone zostało według lub analogicznie do kroków poprzednich - bowiem przynależność danej pary obrazów do danego zbioru określa na obecnym etapie *dataframe* danego zbioru. 
  
   * #### 1.2. Zastosowane programy
   		
     * 1.2.1. Programy umieszczone w niniejszym repozytorium napisane zostały w języku Python 3.7.3.
  
     * 1.2.2. Wykorzystano biblioteki: OpenCV 4.1.0; Numpy 1.16.4; tqdm 4.33.0; oraz standardowe biblioteki języka.
  
     * 1.2.3. Aby wykorzystać programy, które autor opracował do preprocesowania obrazów pisma:
  
        - Należy zainstalować Python oraz wskazane biblioteki (metoda instalacji zależy od systemu operacyjnego użytkownika);
        - Należy pobrać programy z niniejszego repozytorium;
        - Następnie otworzyć - za pomocą danego edytora tekstu - pliki zawierające poszczególne programy i dokonać edycji oznaczonych ścieżek, wskazując foldery gdzie zapisywane mają być obrazy pisma / dataframe (pliki programów zapisać należy w formacie `py`);
        - Plik z danym programem umieścić należy w folderze, w którym znajdują się obrazy pisma, jakie mają zostać przez dany program przetworzone (jest to najprostsza metoda);
        - Folder, który zawiera program i obrazy, otworzyć należy za pomocą terminala / interpretera poleceń (metoda zależna od systemu operacyjnego);
        - Następnie wpisać należy w terminalu komendę `python3 nazwa_programu.py`, stąd wykonany zostanie wskazany program;
        - Gdyby zaistniała taka konieczność: aby przerwać wykonywanie programu wykorzystać można w terminalu kombinację klawiszy `Ctrl + C`, aby zawiesić wykonywanie programu `Ctrl + Z`, aby zaś przywrócić wykonywanie zawieszonego programu wpisać należy komendę `fg` (dla terminalów większości systemów operacyjnych, *e.g.* macOS, Linux).      
  
### 2. Verificational Model

   * #### 2.1. Model
   
     * 2.1.1. Model architecture:
      
        - Siamese CNN - dual path convolutional network, where both paths (left and right path) are two separate ConvNets (either AlexNets or VGG13s), which outputs are flattend, concatenated and then passed to the fully connected layers for binary classification. Inputs to both conv-paths are identical in shape, dataset and preprocessing.
      
        - AlexNet Core Network - {as desc in paper}
        
        - VGG13 Core Network - {as desc in pap except for 2nd pool}
        
        - Fully Connected Layers - either three [4096, 1024, 256] or two [4096, 4096] FCL layers, and one output neuron (sigmoid activation) - for both core network variants;
        
        - Activation - ReLU for all layers, sigmoid for the output neuron;
        
        - Batch Normalization (BN) Layers - applied after ReLU activation;
        
        - Dropout Layers - applied after BN layers;
        
        - Layers order - as suggested in: G. Chen, *et al.*, *Rethinking the Usage of Batch Normalization and Dropout in the Training ofDeep Neural Networks*, arXiv:1905.05928;
        
     * 2.1.2. Language, libraries and API:
        
        - Python3;
        - Numpy (data sequence), Pandas (dataframe), Matplotlib (metrics plot), Pydot and GraphViz (model plot);
        - TensorFlow's implementation of the Keras API (model).
   
     * 2.1.3. Implementation:
       
        - Google Colaboratory - Python 3 Jupyter Notebook, GPU type runtime;
        
        - Time - 170ms/step for Alexnet, XXXms/step for VGG.
        
   * #### 2.2. Training
   
     * 2.2.1. Database:
       
        - Training dataset -
        
        - Validation dataset -
   
     * 2.2.2. Callbacks:
       
        - Tensorboard -
        
        - ModelCheckpoint -
        
        - ReduceLROnPlateau -

     * 2.2.3. Hyperparameters:
      
        - Epochs - 9;
        - Batchsize - 64;
        - Batch normalization - axis=-1, scale=False;
        - Dropout rate - 0.2;
        - Loss - Binary Crossentropy;
        - Metrics - Accuracy; 
        - Optimizer - SGD (Stochastic Gradient Descent);
        - Learning rate - 0.01 (1e-2);
        - LR Decay - 0.0;
        - Momentum - 0.0
        - Nestrov - False.
                
     * 2.2.3. Training evaluation:
       
       | | Core Network | Epochs | FCL | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
       | --- | --- | --- | --- | --- | --- | --- | --- | 
       | Result | AlexNet | 9 | 3 | x | x | x | x |
       | Checkpoint | AlexNet | x | 3 | x | x | x | x |
       | Result | AlexNet | 9 | 2 | x | x | x | x |
       | Checkpoint | AlexNet | x | 2 | x | x | x | x |
        | Result | VGG13 | 9 | 3 | x | x | x | x |
       | Checkpoint | VGG13 | x | 3 | x | x | x | x |
       | Result | VGG13 | 9 | 2 | x | x | x | x |
       | Checkpoint | VGG13 | x | 2 | x | x | x | x |
       
   * #### 2.3. Model evaluation:
   
     * 2.3.1. Database (excluded pairs - both negative and positive - of patches containing identical content):
      
        - CVL Test dataset -
        
        - IAM Test dataset -

     * 2.3.2. Metrics:
        
        - Binary Crossentropy - BN;
        - Accuracy - Acc;
        - True Positive - TP;
        - True Negative - TN;
        - False Positive - FP;
        - False Negative - FN;        
        - Recall `(TP/(TP+FN))` - Rec;
        - Precision `(TP/(TP+FP))` - Pre;
        - Area under the ROC curve - AUC;
      
     * 2.3.3. CVL evaluation:
     
       | Core Network | EofT | FCL | Loss | Acc | TP | TN | FP | FN | Rec | Pre | AUC |
       | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
       | AlexNet | x | 3 | x | x | x | x | x | x | x | x | x | x |
       | AlexNet | x | 2 | x | x | x | x | x | x | x | x | x | x |
       | VGG13 | x | 3 | x | x | x | x | x | x | x | x | x | x |
       | VGG13 | x | 2 | x | x | x | x | x | x | x | x | x | x |
       
       - Epochs of Training (EofT) by the best validation loss result. 
              
     * 2.3.4. IAM evaluation:
       

       | Core Network | EofT | FCL | Loss | Acc | TP | TN | FP | FN | Rec | Pre | AUC |
       | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
       | AlexNet | x | 3 | x | x | x | x | x | x | x | x | x | x |
       | AlexNet | x | 2 | x | x | x | x | x | x | x | x | x | x |
       | VGG13 | x | 3 | x | x | x | x | x | x | x | x | x | x |
       | VGG13 | x | 2 | x | x | x | x | x | x | x | x | x | x |
       
       - Epochs of Training (EofT) by the best validation loss result.
       
