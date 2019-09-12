## Verificational Model

### 1. Preprocessing
  
* #### 1.1. Objective, assumptions, database and steps of preprocessing
   
  * 1.1.1. The objective of author's repository was to introduce a method of image preprocessing for verification of offline handwritten documents authorship by artificial neural networks (through classification of preprocessed image pairs to positive `same author` or negative `different authors` class).
       
  * 1.1.2. Author's assumptions were that:
     
    * Document images will be best processed simultaneously by two separate convolutional neural networks and classified by one multilayer perceptron (fully connected layers);       
    
    * Preprocessing shouldn't drastically affect image quality (*e.g* by image binarization) to preserve most of handwriting features.
            
  * 1.1.3. Database (9455 documents by 2911 writers):
       
    * Dataset of 1604 documents (full page scans) from CVL database (310 writers), by: F. Kleber, S. Fiel, M. Diem, R. Sablatnig, *CVL-Database: An Off-line Database for Writer Retrieval, Writer Identification and Word Spotting*, "In Proc. of the 12th Int. Conference on Document Analysis and Recognition (ICDAR) 2013" 2013, p. 560 - 564;
        
    * Dataset of 4704 documents (full page scans) from CEDAR (Center of Excellence for Document Analysis and Recognition) database (1568 writers), by: S. Srihari, S. Cha, H. Arora, S. Lee, *Individuality of Handwriting*, "Journal of Forensic Sciences" 2002, No. 4 (Vol. 47), p. 1 - 17;
        
    * Dataset of 1539 documents (full page scans) from IAM (offline handwritten documents) database (657 writers), by: U. Marti, H. Bunke, *The IAM-database: An English Sentence Database for Off-line Handwriting Recognition*, "Int'l Journal on Document Analysis and Recognition" 2002, No. 5, p. 39 - 46;
        
    * Dataset of 208 documents (full page scans) from ICDAR 2011 Writer Identification Contest database (26 writers), by: G. Louloudis, N. Stamatopoulos, B. Gatos, *ICDAR 2011 Writer Identification Contest*, "2011 International Conference on Document Analysis and Recognition" 2011, p. 1475 - 1479;
        
    * Dataset of 400 documents (cropped page scans) from ICFHR 2012 Competition on Writer Identification (Challenge 1: Latin/Greek Documents) database (100 writers), by: G. Louloudis, N. Stamatopoulos, B. Gatos, *ICFHR2012 Competition on Writer Identification Challenge 1: Latin/Greek Documents*, "2012 International Conference on Frontiers in Handwriting Recognition" 2012, p. 825 - 830;
        
    * Dataset of 1000 documents (cropped page scans) from ICDAR 2013 Competition on Writer Identification database (250 writers), by: G. Louloudis, N. Stamatopoulos, B. Gatos, *ICDAR 2013 Competition on Writer Identification*, "2013 12th International Conference on Document Analysis and Recognition" 2013, p. 1397 - 1041.
            
  * 1.1.4. Steps of preprocessing:
               
    * `Step_1_Preprocessing.py` - conversion of images (scans of whole documents) to grayscale (scale from black = 0 to white = 255), color inversion, extraction of writing space from images, reduction of extracts dimensions to [1024x1024] pixels, division of extracts into [256x256] pixel patches, conversion from the `tif` to `png` format. Patches which do not contain or contain a small amount of text are skipped by the program because of the arbitrary average pixel value threshold - in any case, patches can be sorted by their size and manually removed on that basis;
            
    * `Step_2_Dataframe.py` - creation of a dataframe (a `csv` file which can be edited in any spreadsheet program, *e.g.* calc / excel) separately for the test and training subset, by combinatorial paring of image names into the positive class (except for identical patches), and a random combinatorial paring of image names into the negative class (the number of possible negative combinations is much greater than the number positive ones, so all positive instances are created first and then the negative instances are randomly combinated until their number is equal to the number of positive instances). Image name pairs and their labels are ordered by rows, according to columns `left convolutional path`, `right convolutional path`, and `label` (labels are determined by the convergence or divergence of author's identifiers - *e.g.* first four digits of an image name in the case of a CVL database). Above method requires that the test and training images will be kept in different directories (during preprocessing). However, it is not necessary to create a `csv` file, *i.e.* it will be created by the program (and if such a file was already created manually, its directory and name has to be indicated in the program code);
        
    * Third step - it is crucial to shuffle rows in dataframes, because `tf.keras.Model.fit_generator` shuffles only the batch order (between epochs), however the method of randomization by rows depends on a given spreadsheet program (most commonly an additional column of random numbers is generated - by a use of `=Rand()` function - and utilized as a Sort Key for Data Sorting in selected columns);
        
    * Fourth step - to create a validation dataframe (utilized only for testing of the model during its training, mostly after each epoch) simply create a `csv` file in any given spreadsheet program, then copy about 10-20% of already randomized instances from the test dataframe (it is most probable that the number of copied positive and negative instances will be effectively equal);
        
    * Fifth step - add column headers to each dataframe, *e.g.* `left convolutional path, right convolutional path, labels` for columns containing names and labels of image pairs.  
        
* #### 1.2. Preprocessing programs
   		
  * 1.2.1. Programming language - Python 3.7.3.
  
  * 1.2.2. Libraries - OpenCV 4.1.0, Numpy 1.16.4, tqdm 4.33.0, and other common python libraries.
  
  * 1.2.3. To utilize preprocessing programs:
  
    * Install Python and listed libraries (installation method depends on user's operating system);
    * Download the repository;
    * Using any given text editor, access programs code and edit image/dataframe save directories paths (save the program files in `py` format); 
    * Place a given preprocessing program file in a directory of images to be preprocessed by that given program (that's the simplest method);
    * Access the directory - which contains a given preprocessing program file and images to be preprocessed - through the terminal / command-line interpreter (method  of access depends on user's operating system);
    * In the terminal type the command `python3 program_name.py` to run the named program;
    * If it were necessary: to force quit (terminate) a running program use a keyboard shortcut `Ctrl + C` in an opened terminal window, or `Ctrl + Z` to suspend a running program, to resume paused run, type the command `fg` (works in terminals of most operating systems, *e.g.* macOS, Linux).
             
  
### 2. Verificational Model

* #### 2.1. Model
   
  * 2.1.1. Model architecture:
      
    * Siamese CNN - dual path convolutional neural network, where both paths (left and right path) are two separate ConvNets (Core Networks), which outputs are concatenated, globally average pooled and then passed to the fully connected layers for binary classification. Inputs to both conv-paths are identical in shape, dataset and preprocessing;
           
    * Core Network - inspired mainly by papers: 
    
      * G. Chen, P. Chen, Y. Shi, C. Hsieh, B. Liao, S. Zhang, *Rethinking the Usage of Batch Normalization and Dropout in the Training of Deep Neural Networks*, arXiv:1905.05928v1 [cs.LG] 2019, p. 1 - 10; 
      
      * M. Lin, Q. Chen, S. Yan, *Network In Network*, arXiv:1312.4400v3 [cs.NE] 2014, p. 1 - 10; 
      
      * A. Krizhevsky, I. Sutskever, G. Hinton, *ImageNet Classification with Deep Convolutional Neural Networks*, "Advances in Neural Information Processing Systems" 2012, No. 1 (Vol. 25), p. 1097 - 1105;
    
    * Globally Average Pooling Layer - applied instead of flattening layer;
    
    * Fully Connected Layers - three dense layers [1024, 512, 256] and one output neuron;
        
    * Activation - ReLU for all layers, sigmoid for the output neuron;
        
    * Batch Normalization (BN) Layers - applied after ReLU activations of convolutional and dense layers;
       
    * Dropout Layers - Gaussian dropout applied before each dense layer.
                    
  * 2.1.2. Language, libraries and framework / API:
        
    * Python3;
    * Numpy (data sequence), Pandas (dataframe), Matplotlib (metrics plot), Pydot and GraphViz (model plot);
    * TensorFlow's implementation of the Keras API (model) v1.14.
   
  * 2.1.3. Implementation:
       
    * Google Colaboratory - (2019) - Python 3 Jupyter Notebook, GPU type runtime (Nvidia Tesla K80), 290ms/step (27732 steps per epoch);
        
    * Google Kaggle - (2019) - Python 3 Jupyter Notebook, GPU type runtime (Nvidia Telsa P100), 65ms/step (27732 steps per epoch).
        
* #### 2.2. Model training
   
  * 2.2.1. Database:
       
    * Training dataset - CVL database subset of 1415 document images by 283 writers - 443704 image pairs (221852 positive and 221852 negative instances);
        
    * Validation dataset - CVL database subset of 189 document images by 27 writers - 12478 image pairs (6300 positive and 6178 negative instances).
          
  * 2.2.2. Callbacks:
     
    * CSVLogger - streams epoch results to a csv file;
       
    * Tensorboard - logs events for TensorBoard visualization tool;
        
    * Model Checkpoint - saves the model after every epoch of validation loss improvement;
        
    * Reduce LR On Plateau - reduces learning rate by a given factor after every epoch of validation loss deterioration.

  * 2.2.3. Hyperparameters:
      
    * Epochs - 8 (two runs, 4 epochs each);
    * Batchsize - 16;
    * Dropout rate - 0.5;
    * Loss - Binary Crossentropy;
    * Metrics - Accuracy; 
    * Optimizer - Adam (Adaptive Moment Estimation):
      * Learning rate (alpha) - epochs 1/4 - 0.001;
      * Learning rate (alpha) - epochs 5/8 - 0.0001;
      * Beta_1 , beta_2, epsilon - as recommended by: D. Kingma, J. Ba, *Adam: A Method for Stochastic Optimization*,  arXiv:1412.6980v9 [cs.LG] 2017, p. 2.
    * ReduceLROnPlateau - factor 0.1. 
        
  * 2.2.4. Training:
     
    | Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate Reductions |
    | --- | --- | --- | --- | --- |  --- |
    | 1 | 0.4500 | 0.7960 | 0.3918 | 0.8279 | None |
    | 2 | 0.2830 | 0.8916 | 0.3544 | 0.8529 | None |
    | 3 | 0.2125 | 0.9229 | 0.2547 | 0.9007 | None |
    | 4 | 0.1828 | 0.9353 | 0.2458 | 0.9050 | Manual LR reduction to 0.0001 (1e-4) |
    | **5** | 0.1348 | 0.9540 | **0.2320** | **0.9110** | None |
    | 6 | 0.1165 | 0.9614 | 0.2336 | 0.9140 | Callback LR reduction to 0.00001 (1e-5) |
    | 7 | 0.1061 | 0.9653 | 0.2328 | 0.9134 | Callback LR reduction to 0.000001 (1e-6) |
    | 8 | 0.1037 | 0.9657 | 0.2347 | 0.9130 | Callback LR reduction to 0.0000001 (1e-7) |
     
![loss](https://github.com/Ma-Marcinowski/Verificational-Model/blob/master/Model_Loss.png "Model Loss") ![acc](https://github.com/Ma-Marcinowski/Verificational-Model/blob/master/Model_Accuracy.png "Model Accuracy")

* #### 2.3. Model evaluation
   
  * 2.3.1. Database:
      
    * CVL database:
      
      * Test subset of 189 document images by 27 writers;
      
      * Custom soft criterion (excluded pairs of identical patches) - 82476 image pairs (41238 positive and 41238 negative instances);
        
    * IAM database:
      
      * Whole database as a test set of 1539 document images by 657 writers;
      
      * Custom soft criterion (excluded pairs of identical patches) - x image pairs (x positive and x negative instances).
      
      * Custom hard criterion (excluded both negative and positive pairs of documents containing identical sample text) - x image pairs (x positive and x negative instances).
        
  * 2.3.2. Metrics:
        
    * Binary Crossentropy - BN;
    * Accuracy - Acc;
    * True Positive Rate `(TP/P)` - TPR;
    * True Negative Rate `(TN/N)` - TNR;
    * False Positive Rate `(FP/N)` - FPR;
    * False Negative Rate `(FN/P)` - FNR;         
    * Recall `(TP/(TP+FN))` - Rec;
    * Precision `(TP/(TP+FP))` - Pre;
    * Area under the ROC curve - AUC.
      
  * 2.3.3. CVL evaluation (epochs of model training - EofT - by the best validation result):
     
  | Criterion | EofT | Loss | Acc | TPR | TNR | FPR | FNR | Rec | Pre | AUC |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | Soft | 5 | 0.2331 | 0.9124 | 0.9207 | 0.9040 | 0.0959 | 0.0792 | 0.9207 | 0.9056 | 0.9676 |
              
  * 2.3.4. IAM evaluation (epochs of model training - EofT - by the best validation result):
       
  | Criterion | EofT | Loss | Acc | TPR | TNR | FPR | FNR | Rec | Pre | AUC |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | Soft | 5 | x | x | x | x | x | x | x | x | x |
  | Hard | 5 | x | x | x | x | x | x | x | x | x |
