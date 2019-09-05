## Verificational Model

### 1. Preprocessing
  
   * #### 1.1. Objective, assumptions, database and steps of preprocessing
   
     * 1.1.1. The objective of author's repository was to introduce a method of image preprocessing for verification of offline handwritten documents authorship by artificial neural network (through classification of preprocessed image pairs to positive `same author` or negative `different authors` class).
       
     * 1.1.2. Author's assumptions were that:
     
        - Document images ought to be processed simultaneously by two separate convolutional neural networks and classified by one multilayer perceptron (fully connected layers);       
        - Preprocessing shouldn't affect image quality to preserve most of handwriting features.
            
     * 1.1.3. Database:
       
        - Subsets of 1415 training and 189 test documents (full page scans) from CVL database (F. Kleber, S. Fiel, M. Diem, R. Sablatnig, *CVL-Database: An Off-line Database for Writer Retrieval, Writer Identification and Word Spotting*, "In Proc. of the 12th Int. Conference on Document Analysis and Recognition (ICDAR) 2013" 2013, p. 560 - 564). 
            
     * 1.1.4. Steps of preprocessing:
               
        - `Step_1_Preprocessing.py` - conversion of images (scans of whole documents) to grayscale (scale from black = 0 to white = 255), color inversion, extraction of writing space from images, reduction of extracts dimensions to [1024x1024] pixels, division of extracts into [256x256] pixel patches, conversion from the `tif` to `png` format. Patches which do not contain or contain a small amount of text are skipped by the program because of the arbitrary average pixel value threshold - in any case, patches can be sorted by their size and manually removed on that basis;
            
        - `Step_2_Dataframe.py` - creation of a dataframe (a `csv` file which can be edited in any spreadsheet program, *e.g.* calc / excel) separately for the test and training subset, by combinatorial paring of image names into the positive class, and a random combinatorial paring of image names into the negative class (the number of possible negative combinations is much greater than the number positive ones, so all positive instances are created first and then the negative instances are randomly combinated until their number is equal to the number of positive instances). Image name pairs and their labels are ordered by rows, according to columns `left convolutional path`, `right convolutional path`, and `label` (labels are determined by the convergence or divergence of identifiers - *i.e.* first four digits of the image name which denote its author). Above method requires that test and training images are kept in different directories. However, it is not necessary to create a `csv` file, *i.e.* it will be created by the program (and if such a file was already created manually, please indicate its directory and name in the program code);
        
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
        
        - VGG13 Core Network - {as desc in pap except for 2nd pool}. Hovever the output size of the first convolutional block (1st and 2nd conv layer followed by 1st max pooling layer) is reduced by pooling of size [3x3] and stride [3x3] - as opposed to vanilla 1st max pooling layer where pool size is [2x2] and stride is [2x2]. Therefore (flattend and concatenated outputs of core nets) input to FCL is of size [none, 25000] as opposed to [none, 65000].
        
        - Fully Connected Layers - three [4096, 1024, 256] FC layers, and one output neuron (sigmoid activation) AlexNet core network;
        
        - Activation - ReLU for all layers, sigmoid for the output neuron;
        
        - Batch Normalization (BN) Layers - applied after ReLU activations of convolutional layers;
        
        - Dropout Layers - applied after dense FCL layers;
               
     * 2.1.2. Language, libraries and framework / API:
        
        - Python3;
        - Numpy (data sequence), Pandas (dataframe), Matplotlib (metrics plot), Pydot and GraphViz (model plot);
        - TensorFlow's implementation of the Keras API (model).
   
     * 2.1.3. Implementation:
       
        - Google Colaboratory - Python 3 Jupyter Notebook, GPU type runtime;
        
        - Time - 500ms/step for Alexnet, XXXms/step for VGG. ???
        
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
        - Batch normalization - Scale=False;
        - Dropout rate - 0.2;
        - Loss - Binary Crossentropy;
        - Metrics - Accuracy; 
        - Optimizer - SGD (Stochastic Gradient Descent);
        - Learning rate - 0.01;
        - LR Decay - 0.0;
        - Momentum - 0.0;
        - Nestrov - False;
        - Shuffle - True.
                
     * 2.2.4. Training:
       
       | | Core Network | Epochs | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
       | --- | --- | --- | --- | --- | --- | --- | 
       | Result | AlexNet | 9 |  x | x | x | x |
       | Checkpoint | AlexNet | x | x | x | x | x |     
       | Result | VGG13 | 9 | x | x | x | x |
       | Checkpoint | VGG13 | x | x | x | x | x |
       
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
     
       | Core Network | EofT | Loss | Acc | TP | TN | FP | FN | Rec | Pre | AUC |
       | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
       | AlexNet | x | x | x | x | x | x | x | x | x | x | x |
       | VGG13 | x | x | x | x | x | x | x | x | x | x | x |
              
       - Epochs of Training (EofT) by the best validation loss result. 
              
     * 2.3.4. IAM evaluation:
       

       | Core Network | EofT | Loss | Acc | TP | TN | FP | FN | Rec | Pre | AUC |
       | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
       | AlexNet | x | x | x | x | x | x | x | x | x | x | x |
       | VGG13 | x | x | x | x | x | x | x | x | x | x | x |
       
       - Epochs of Training (EofT) by the best validation loss result.
       
