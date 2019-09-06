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
        
        - Third step - it is crucial to shuffle rows in dataframes, because `tf.keras.Model.fit_generator` shuffles only the batch order (between epochs), however the method of randomization by rows depends on a given spreadsheet program (most commonly an additional column of random numbers is generated - by a use of `=Rand()` function - and utilized as a Sort Key for Data Sorting in selected columns);
        
        - Fourth step - to create a validation dataframe (utilized only for testing of the model during its training, mostly after each epoch) simply create a `csv` file in any given spreadsheet program, then copy about 10-20% of already randomized instances from the test dataframe (it is most probable that the number of copied positive and negative instances will be effectively equal);
        
        - Fifth step - add column headers to each dataframe, *e.g.* `left convolutional path, right convolutional path, labels` for columns containing names and labels of image pairs.  
        
   * #### 1.2. Preprocessing programs:
   		
     * 1.2.1. Programming language - Python 3.7.3.
  
     * 1.2.2. Libraries - OpenCV 4.1.0, Numpy 1.16.4, tqdm 4.33.0, and other common python libraries.
  
     * 1.2.3. To utilize preprocessing programs:
  
        - Install Python and listed libraries (installation method depends on user's operating system);
        - Download the repository;
        - Using any given text editor, access programs code and edit image/dataframe save directories paths (save the program files in `py` format); 
        - Place a given preprocessing program file in a directory of images to be preprocessed by that given program (that's the simplest method);
        - Access the directory - which contains a given preprocessing program file and images to be preprocessed - through the terminal / command-line interpreter (method  of access depends on user's operating system);
        - In the terminal type the command `python3 program_name.py` to run the named program;
        - If it were necessary: to force quit (terminate) a running program use a keyboard shortcut `Ctrl + C` in an opened terminal window, or `Ctrl + Z` to suspend the running program, to resume the paused run type the command `fg` (works in terminals of most operating systems, *e.g.* macOS, Linux).
             
  
### 2. Verificational Model

   * #### 2.1. Model
   
     * 2.1.1. Model architecture v1.0:
      
        - Siamese CNN - dual path convolutional network, where both paths (left and right path) are two separate ConvNets (of a given variant), which outputs are flattend, concatenated and then passed to the fully connected layers for binary classification. Inputs to both conv-paths are identical in shape, dataset and preprocessing;
           
        - AlexNet Core Network variant 1.0 - {as desc in paper};
        
        - VGG13 Core Network variant 1.0 - {as desc in paper}. However, size of the output of the network is reduced by stride [2x2] on the first convolutional layer - as opposed to vanilla stride [1x1] on the first conv layer. Therefore (flattend and concatenated outputs of VGG core nets) input to FCL is of size [none, 16384] as opposed to [none, 65536];
        
        - Fully Connected Layers - three FC layers [4096, 1024, 256] and one output neuron (sigmoid activation) for both core network variants;
        
        - Activation - ReLU for all layers, sigmoid for the output neuron;
        
        - Batch Normalization (BN) Layers - applied after ReLU activations of convolutional layers;
        
        - Dropout Layers - applied before each dense FCL layer and after each MaxPooling Layer;
        
        - L2 regularizer - applied on all FCL layers. 
     
     * 2.1.2. Model architecture v2.0:
     
       - Siamese CNN - dual path convolutional network, where both paths (left and right path) are two separate ConvNets (of a given variant), which outputs are globally average pooled and passed to the output neuron for binary classification. Inputs to both conv-paths are identical in shape, dataset and preprocessing;
           
        - AlexNet Core Network variant 2.0 - {as desc in paper};
        
        - Global Average Pooling Layer - applied instead of fully connected layers. 
        
        - Activation - ReLU for all layers, sigmoid for the output neuron;
        
        - Batch Normalization (BN) Layers - applied after ReLU activations of convolutional layers;
               
     * 2.1.3. Language, libraries and framework / API:
        
        - Python3;
        - Numpy (data sequence), Pandas (dataframe), Matplotlib (metrics plot), Pydot and GraphViz (model plot);
        - TensorFlow's implementation of the Keras API (model).
   
     * 2.1.4. Implementation:
       
        - Google Colaboratory - Python 3 Jupyter Notebook, GPU type runtime - 2019;
        
        - Time - 300ms/step for Alexnet v1.0, 000ms/step for VGG v1.0, 450ms/step for Alexnet v2.0.
        
   * #### 2.2. Training
   
     * 2.2.1. Database:
       
        - Training dataset -
        
        - Validation dataset -
          
     * 2.2.2. Callbacks:
       
        - Tensorboard -
        
        - ModelCheckpoint -
        
        - ReduceLROnPlateau -

     * 2.2.3. Hyperparameters v1.0:
      
        - Epochs - 6; 
        - Batchsize - 16;
        - Dropout rate - 0.2;
        - Loss - Binary Crossentropy;
        - Metrics - Accuracy; 
        - Optimizer - SGD (Stochastic Gradient Descent);
        - Learning rate - 0.01;
        - LR Decay - 0.0;
        - Momentum - 0.0;
        - Nestrov - False.
                
     * 2.2.3. Hyperparameters v2.0:
                
        - Epochs - 12; 
        - Batchsize - 64;
        - Dropout rate - 0.0;
        - Loss - Binary Crossentropy;
        - Metrics - Accuracy; 
        - Optimizer - SGDM (Stochastic Gradient Descent with Momentum);
        - Learning rate - 0.01;
        - LR Decay - 0.0005 (5e-4);
        - Momentum - 0.9;
        - Nestrov - False.
                
     * 2.2.4. Training:
       
       | | Core Network | Epochs | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
       | --- | --- | --- | --- | --- | --- | --- | 
       | Result | AlexNet v1.0 | x |  x | x | x | x |
       | Checkpoint | AlexNet v1.0 | x | x | x | x | x |     
       | Result | VGG13 v1.0 | x | x | x | x | x |
       | Checkpoint | VGG13 v1.0 | x | x | x | x | x |
       | Result | AlexNet v2.0 | x |  x | x | x | x |
       | Checkpoint | AlexNet v2.0 | x | x | x | x | x |
       
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
       | AlexNet v1.0 | x | x | x | x | x | x | x | x | x | x | x |
       | VGG13 v1.0 | x | x | x | x | x | x | x | x | x | x | x |
       | AlexNet v2.0 | x | x | x | x | x | x | x | x | x | x | x |
       
       - Epochs of Training (EofT) by the best validation result. 
              
     * 2.3.4. IAM evaluation:
       

       | Core Network | EofT | Loss | Acc | TP | TN | FP | FN | Rec | Pre | AUC |
       | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
       | AlexNet v1.0 | x | x | x | x | x | x | x | x | x | x | x |
       | VGG13 v1.0 | x | x | x | x | x | x | x | x | x | x | x |
       | AlexNet v2.0 | x | x | x | x | x | x | x | x | x | x | x |
       
       - Epochs of Training (EofT) by the best validation result.
       
