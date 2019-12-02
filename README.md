## Verificational Model

### 0. Introduction

* #### 0.1. The objective of author's repository was to introduce multiple varying methods for verification of offline handwritten documents authorship by artificial neural networks (through classification of preprocessed image pairs to positive `same author` or negative `different authors` class). The purpose of such methods, models and experiments as described below, was to create an empirical background for scientific analysis of machine learning tools developed in the field of computational forensics.

* #### 0.2. Author's original assumptions were that:
     
    * Document images will be best processed simultaneously by two separate convolutional neural networks and classified by one multilayer perceptron (fully connected layers);       
    
    * Preprocessing shouldn't drastically affect image quality (e.g. by image binarization) to preserve most of handwriting features.
    
    * The number of positive and negative class instances (generated for the purpose of model training / testing) ought to be equal.

* #### 0.3. Versioning

   * 0.3.1 Experiment identifiers (e.g. vX.Y.Z):

      * X indicates a model version;
      * Y indicates a method of preprocessing;
      * Z indicates any extra variation of a given X.Y base combination.

   * 0.3.2 Model version identifiers (e.g. vX):

      * X indicates a model version (as already stated in 0.3.1).

   * 0.3.3 Preprocessing method identifiers (e.g. v0.Y):
    
      * X is zeroed and Z skipped to avoid any confusion of model versions and preprocessing methods; 
      * Y indicates a method of preprocessing (as already stated in 0.3.1).

* #### 0.4. Keywords:

    * Computational, statistical, probabilistic; 
    
    * Forensics, criminalistics, analysys, examination;
    
    * Handwriting, signatures, documents;
    
    * Neural, networks, deep, machine, learning, artificial, intelligence, ANN, AI.

* #### 0.5. Database (9455 documents by 2911 writers):
       
    * Dataset of 1604 documents (full page scans) from CVL database (310 writers), by: F. Kleber, S. Fiel, M. Diem, R. Sablatnig, *CVL-Database: An Off-line Database for Writer Retrieval, Writer Identification and Word Spotting*, "In Proc. of the 12th Int. Conference on Document Analysis and Recognition (ICDAR) 2013" 2013, p. 560 - 564;
        
    * Dataset of 4704 documents (full page scans) from CEDAR (Center of Excellence for Document Analysis and Recognition) database (1568 writers), by: S. Srihari, S. Cha, H. Arora, S. Lee, *Individuality of Handwriting*, "Journal of Forensic Sciences" 2002, No. 4 (Vol. 47), p. 1 - 17;
        
    * Dataset of 1539 documents (full page scans) from IAM (offline handwritten documents) database (657 writers), by: U. Marti, H. Bunke, *The IAM-database: An English Sentence Database for Off-line Handwriting Recognition*, "Int'l Journal on Document Analysis and Recognition" 2002, No. 5, p. 39 - 46;
        
    * Dataset of 208 documents (full page scans) from ICDAR 2011 Writer Identification Contest database (26 writers), by: G. Louloudis, N. Stamatopoulos, B. Gatos, *ICDAR 2011 Writer Identification Contest*, "2011 International Conference on Document Analysis and Recognition" 2011, p. 1475 - 1479;
        
    * Dataset of 400 documents (cropped page scans) from ICFHR 2012 Competition on Writer Identification (Challenge 1: Latin/Greek Documents) database (100 writers), by: G. Louloudis, N. Stamatopoulos, B. Gatos, *ICFHR2012 Competition on Writer Identification Challenge 1: Latin/Greek Documents*, "2012 International Conference on Frontiers in Handwriting Recognition" 2012, p. 825 - 830;
        
    * Dataset of 1000 documents (cropped page scans) from ICDAR 2013 Competition on Writer Identification database (250 writers), by: G. Louloudis, N. Stamatopoulos, B. Gatos, *ICDAR 2013 Competition on Writer Identification*, "2013 12th International Conference on Document Analysis and Recognition" 2013, p. 1397 - 1041.

### 1.1. Preprocessing v0.1 (CVL database, grayscaled images)
              
* #### 1.1.1. Steps of preprocessing in the case of CVL database:
               
    * Step one `CVL_Images_v0.1.py` - conversion of images (scans of whole documents) to grayscale (scale from black = 0 to white = 255), color inversion, extraction of writing space from images, reduction of extracts dimensions to [1024x1024] pixels, division of extracts into [256x256] pixel patches, conversion from the `tif` to `png` format. Patches which do not contain or contain a small amount of text are omitted by the program, based on the the arbitrary average pixel value threshold - in any case, patches can be sorted by their size and manually removed on that basis;
            
    * Step two `Dataframe_v0.1.py` - creation of a dataframe (i.e. of a `csv` file that can be edited in any spreadsheet program, e.g. calc / excel) separately for the test and training subsets, by combinatorial pairing of image names into the positive class, and a random combinatorial pairing of image names into the negative class (the number of possible negative combinations is much greater than the number positive ones, so all positive instances are created first and then the negative instances are randomly combinated until their number is equal to the number of positive instances). It has to be noted however, that for any given positive `xy` pair, also the reverse pair `yx` will be created (in the case of negative class, there is a possibility that for any randomly generated `xy` also `yx` will be randomly generated). Image name pairs and their labels are ordered by rows, according to columns `left convolutional path`, `right convolutional path`, and `label` (labels are determined by the convergence or divergence of author's identifiers - e.g. first four digits of a raw image name in the case of a CVL database). Above method requires that the test and training images are kept in different directories during their preprocessing. However, it is not necessary to create manually any of dataframe `csv` files, i.e. they will be created by the program (and if any such a file was already created manually, its directory and name has to be indicated in the program code). Validation dataframe (utilized only for testing of the model during its training, generally after every epoch) is also created, by random sampling of the test dataframe instances (fratcion of which to pull has to be specified - usually 0.1 / 0.2 is enough for validation purposes). Due to the randomness of sampling, it is most probable that the number of sampled positive and negative instances will be effectively equal.
               
* #### 1.1.2. Preprocessing programs
   		
  * 1.1.2.1. Programming language - Python 3.7.3.
  
  * 1.1.2.2. Libraries - OpenCV 4.1.0, Numpy 1.16.4, tqdm 4.33.0, and other common python libraries.
  
  * 1.1.2.3. To utilize preprocessing programs:
  
    * Install Python and listed libraries (installation method depends on user's operating system);
    * Download the repository;
    * Using any given text editor, access programs code and edit image/dataframe input/output paths (save the program files in a `py` format); 
    * Access the directory - which contains a given preprocessing program file - through the terminal / command-line interpreter (method  of access depends on user's operating system);
    * In the terminal type the command `python3 program_name.py` to run the named program;
    * If it were necessary: to force quit (terminate) a running program use a keyboard shortcut `Ctrl + C` in an opened terminal window, or `Ctrl + Z` to suspend a running program, then to resume a paused run, type the command `fg` (works in terminals of most operating systems, e.g. macOS, Linux).

### 1.2. Preprocessing v0.2 (CVL database, binarized images)

* #### 1.2.1. In the case of CVL database, method of preprocessing is exactly the same as v0.1, except for:

    * Image binarization (Otsu's method), hence higher threshold of mean pixel value is applied;
    
    * Extraction window is slightly shifted to better fit the writting space, hence greater number of image patches is passed through the mean pixel value threshold.

* #### 1.2.2. Dataframes are generated exatly the same way as in the case of v0.1.

### 1.3. Preprocessing v0.3 (CVL and IAM database, binarized images)

* #### 1.3.1. In the case of CVL database, method of preprocessing is exactly the same as v0.2.

* #### 1.3.2. In the case of IAM database, method of preprocessing is exactly the same as in the case of CVL database (extraction window is slightly shifted), however following additional steps have been applied:
      
    * Before preprocessing - two daraframes of raw IAM images were manually created (consisting of images paths, authors ids and forms ids), one for testset and one for trainset, both were uploaded into the repository (vide `/Examples/Dataframes/` folder);
    
    * During preprocessing - before Otsu's binarization is applied, background noises are removed from the images by thresholding of pixel values below 55 to 0.

* #### 1.3.3. Dataframes are generated exatly the same way as in the case of v0.1 and v0.2, except for the optional split of training dataframe into a given number of smaller equal size training dataframes, due to the sheer number of training image pairs (3.5 million), to assure a better control over the training process.

### 1.4. Preprocessing v0.4 (CVL and IAM database, binarized and denoised images)

* #### 1.4.1. Exactly the same as v0.3, however:
     
     * Threshold of the function thresholding to zero was lowered from 55 to 25;
     
     * Thresholding to zero is applied to both IAM and CVL images, hence the model shouldn't differentiate
 between image sources (*however it still does, as may be concluded from the results of negative criterion test, vide 3.10.3.*).
 
### 1.5. Preprocessing v0.5 (CVL and IAM database, grayscaled and noised images)

* #### 1.5.1. Analogous to v0.4, however:

     * Images are grayscaled instead of binarized;
     
     * To minimize overfitting and perturbations (due to the noise present in some cases of IAM images), the threshold of a function thresholding to zero is lowered to 15, and slight noise is added to all images (both CVL and IAM);
     
     * Because thresholding of empty patches by mean pixel value proved ineffective over time, another more subtle method is applied, i.e. any given patch is multiplied by a filter matrix (vide `/Examples/Preprocessing-Filters/` folder) and accepted if the sum of all it's elements is higher than a zero. It has to be noted that ultimately five filters are applied, therefore any given patch effectively has to pass five thresholds;
     
     * No cross-databases pairs are generated, i.e. negative instances of inter databases image pairs, such that for any given `xy` pair, an `x` belongs to CVL testset and `y` to IAM testset (or vice versa).     
     
### 1.6. Preprocessing v0.6 (CVL and IAM database, grayscaled and noised images)

* #### 1.6.1. Exactly the same as v0.5, however dataframes are generated differently:

    * No reverse pairs are created (neither positive nor negative), e.g. if a pair `xy` was already generated, then a pair `yx` is omitted;
    
    * Train and validation dataframes are still created under the assumption that the number of positive and negative instances ought to be equal; 
    
    * For the purpose of standard model testing, a validation dataframe is generated to the extent of possible positive instances, where the number of positive and negative instances is equal, and divided into `n` smaller validation dataframes utilized for the purpuose of rough testing between epochs;
    
    * The test dataframe is generated for the purpose of *combined evaluation* (vide 3.16.1. Method of combined evaluation), hence all possible negative and positive instances are created (except for pair reverses).
    
### 1.7. Preprocessing v0.7 (CVL and IAM database, ??? images)

* #### 1.7.1. Exactly the same as v0.6, however:

    * Neither test nor train dataframes are created under the assumption that the number of positive and negative instances ought to be equal; 
    
    * Hence for the purpose of training, a dataframe of all possible negative and positive instances is generated (except for pair reverses) and divided into `m` equal parts;
    
* #### 1.7.2. Therefore:

    * Validation dataframes are generated the same way as in the case of v0.6;
     
    * The test dataframe is generated for the purpose of *combined evaluation* exactly as in the case of v0.6.
  
### 2. Verificational Model v1

* #### 2.1. Model v1.1.0
   
  * 2.1.1. Model architecture:
      
    * Siamese CNN - dual path convolutional neural network, where both paths (left and right path) are two separate ConvNets (Core Networks), which outputs are concatenated, globally average pooled and then passed to the fully connected layers for binary classification. Inputs to both conv-paths are identical in shape, dataset and preprocessing;
           
    * Core Network - inspired mainly by papers: 
    
      * G. Chen, P. Chen, Y. Shi, C. Hsieh, B. Liao, S. Zhang, *Rethinking the Usage of Batch Normalization and Dropout in the Training of Deep Neural Networks*, arXiv:1905.05928v1 [cs.LG] 2019, p. 1 - 10; 
      
      * M. Lin, Q. Chen, S. Yan, *Network In Network*, arXiv:1312.4400v3 [cs.NE] 2014, p. 1 - 10; 
      
      * A. Krizhevsky, I. Sutskever, G. Hinton, *ImageNet Classification with Deep Convolutional Neural Networks*, "Advances in Neural Information Processing Systems" 2012, No. 1 (Vol. 25), p. 1097 - 1105;
    
    * Globally Average Pooling Layer - applied instead of flattening layers;
    
    * Fully Connected Layers - three dense layers [1024, 512, 256] and one output neuron;
        
    * Activation - ReLU for all layers, sigmoid for the output neuron;
        
    * Batch Normalization (BN) Layers - applied after ReLU activations of convolutional and dense layers;
       
    * Dropout Layers - Gaussian dropout applied before each dense layer.
                    
  * 2.1.2. Language, libraries and framework / API:
        
    * Python3;
    * Numpy (data sequence), Pandas (dataframe), Matplotlib (metrics plot), Pydot and GraphViz (model plot);
    * TensorFlow's implementation of the Keras API (model) v1.14.
   
  * 2.1.3. Implementation:
       
    * Google Colaboratory - (2019) - Python 3 Jupyter Notebook, GPU type runtime (Nvidia Tesla K80), 290ms/step (27732 steps per epoch) in case of model v1.1.0 training on [256x256] patches;
        
    * Google Kaggle - (2019) - Python 3 Jupyter Notebook, GPU type runtime (Nvidia Telsa P100), 65ms/step (27732 steps per epoch) in case of model v1.1.0 training on [256x256] patches.
        
* #### 2.2. Model v1.1.0 training on [256x256] patches
   
  * 2.2.1. Database:
  
    * Preprocessing v0.1
  
    * Training dataset - CVL database subset of 1415 document images by 283 writers - 443704 image pairs (equal number of positive and negative instances);
        
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
      * Initial learning rate (alpha) - 0.001 (1e-3);
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
     
![loss](https://github.com/Ma-Marcinowski/Verificational-Model/blob/master/Plots/VM_v1.1.0_Loss.png "Model Loss") ![acc](https://github.com/Ma-Marcinowski/Verificational-Model/blob/master/Plots/VM_v1.1.0_Accuracy.png "Model Accuracy")

* #### 2.3. Model v1.1.0 evaluation on [256x256] patches
   
  * 2.3.1. Database:
      
    * CVL database:
      
      * Test subset of 189 document images by 27 writers;
      
      * 82476 image pairs (equal number of positive and negative instances);
      
      * Hard criterion - excluded documents containing the same samlpe text as train documents (*ergo* included documents containing only samlpe texts no. 7 and 8) - 7572 image pairs (equal number of positive and negative instances).
      
    * IAM database:
      
      * Whole database as a test set of 1539 document images by 657 writers;
      
      * 589274 image pairs only, due to a high background noise and thefore a threshold of mean pixel value >= 16 (equal number of positive and negative instances);
      
      * 590000 roughly denoised (by thresholding of pixel values lower than 55 to 0) images pairs (equal number of positive and negative instances).
        
  * 2.3.2. Metrics:
        
    * Binary Crossentropy - Loss;
    * Accuracy - Acc;
    * True Positive Rate / Sensitivity - `(TP/P)` - TPR;
    * True Negative Rate / Specificity - `(TN/N)` - TNR;
    * False Positive Rate - `(FP/N)` - FPR;
    * False Negative Rate - `(FN/P)` - FNR;
    * Positive Predictive Value - `(TP/(TP+FP))` - PPV;
    * Negative Predictive Value - `(TN/(TN+FN))` - NPV;
    * Area under the ROC curve - AUC.
      
  * 2.3.3. CVL evaluation:
    
    | Criterion | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | None | 5 | 0.2331 | 0.9124 | 0.9207 | 0.9040 | 0.0959 | 0.0792 | 0.9056 | 0.9194 | 0.9676 |
    | **Hard** | 5 | **0.1795** | **0.9370** | 0.9617 | 0.9123 | 0.0876 | 0.0382 | 0.9164 | 0.9597 | 0.9781 |
              
    * Epochs of model training - EofT - by the best validation loss result;
              
  * 2.3.4. IAM evaluation:
    
    | Denoised | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | **False** | 5 | 265.3726 | 0.5047 | **0.0113** | **0.9979** | 0.0020 | 0.9886 | 0.8478 | 0.5023 | 0.5058 |
    | True | 5 | 0.3629 | 0.8835 | 0.9661 | 0.8009 | 0.1990 | 0.0338 | 0.8291 | 0.9594 | 0.9603 |
    
    * Epochs of model training - EofT - by the best validation loss result;

* #### 2.4. Model v1.1.1 training on [512x512] patches
  
  * 2.4.1. Database:
  
    * Preprocessing v0.1 ([1024x1024] extracts split into 4 patches instead of 16);
       
    * Training dataset - CVL database subset of 1415 document images by 283 writers - 59868 image pairs (equal number of positive and negative instances);
        
    * Validation dataset - CVL database subset of 189 document images by 27 writers - 2436 image pairs (1201 positive and 1235 negative instances).

  * 2.4.2. Callbacks: 
  
    * Vide 2.2.2. Callbacks.
  
  * 2.4.3. Hyperparameters: 
    
    * Vide 2.2.3. Hyperparameters.
   
  * 2.2.4. Training:
     
    | Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate Reductions |
    | --- | --- | --- | --- | --- |  --- |
    | 1 | 0.5774 | 0.7126 | 0.5133 | 0.7389 | None |
    | 2 | 0.5340 | 0.7412 | 0.4941 | 0.7525 | None |
    | 3 | 0.4665 | 0.7849 | 0.3565 | 0.8432 | None |
    | 4 | 0.3943 | 0.8322 | 0.4770 | 0.7570 | None |
    | 5 | 0.3507 | 0.8546 | 0.2973 | 0.8822 | None |
    | 6 | 0.3354 | 0.8631 | 0.3052 | 0.8744 | Callback LR reduction to 0.00001 (1e-5) |
    | 7 | 0.3359 | 0.8638 | **0.2973** | **0.8838** | None |
    | 8 | 0.3336 | 0.8646 | 0.3033 | 0.8785 | Callback LR reduction to 0.000001 (1e-6) |

* #### 2.5. Model v1.1.2 training on [1024x1024] extracts
 
  * 2.5.1. Database:
  
    * Preprocessing v0.1 (except for split of extracts into patches);
       
    * Training dataset - CVL database subset of 1415 document images by 283 writers - 14150 image pairs (equal number of positive and negative instances;
        
    * Validation dataset - CVL database subset of 189 document images by 27 writers - 646 image pairs (319 positive and 327 negative instances).

  * 2.5.2. Callbacks: 
  
    * Vide 2.2.2. Callbacks.
  
  * 2.5.3. Hyperparameters: 
    
    * Vide 2.2.3. Hyperparameters;
    * Epochs - 3;
    * Batchsize - 8.
   
  * 2.5.4. Training:
     
    | Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate Reductions |
    | --- | --- | --- | --- | --- |  --- |
    | 1 | 0.7105 | 0.6174 | **0.4833** | **0.7724** | None |
    | 2 | 0.6059 | 0.6739 | 0.5242 | 0.7477 | Callback LR reduction to 0.0001 (1e-4) |
    | 3 | 0.5786 | 0.6973 | 0.5527 | 0.7183 | Callback LR reduction to 0.00001 (1e-5) |

### 3. Verificational Model v2

* #### 3.1. Model v2.1.0
  
  * 3.1.1. Model architecture is the same as model v1 architecture, however:
  
    * Globally average pooling layer is not applied to the output of concatenation layer, but at the output level of each core network, i.e. instead of the output max pooling layers (therefore flattening layer is again unncesessary);
    
    * Kernel strides and sizes are slightly higher - in the case of first and second level convolutional layers - compared to the v1 model;
    
    * Fully connected layers were not modified, however the output neuron of those layers is not utilized as an output neuron of the network, also ReLU activation is applied to that neuron instead of the sigmoid;
    
    * The output neuron, to which sigmoid activation is applied, is stacked on top of the network and performs the task of classification based on the three following inputs - an output signal from the FCL output neuron, Euclidean distance and Cosine distance of a given feature vectors pair (extracted by convolutional core networks from a given image pair). 
    
  * 3.1.2. Language, libraries and framework / API:
        
    * Vide 2.1.2.
   
  * 3.1.3. Implementation:
       
    * Google Colaboratory - (2019) - Python 3 Jupyter Notebook, GPU type runtime (Nvidia Tesla K80), 250ms/step (27732 steps per epoch) in case of model v2.1.0 training on [256x256] patches.    
  
* #### 3.2. Model v2.1.0 training on [256x256] patches
 
  * 3.2.1. Database:
  
    * Vide 2.2.1. Database.
 
  * 3.2.2. Callbacks: 
  
    * Vide 2.2.2. Callbacks.
  
  * 3.2.3. Hyperparameters: 
    
    * Vide 2.2.3. Hyperparameters;
    * Epochs - 6 (six separate runs);
    * Learning rate - initial 0.001 (1e-3), then manually adjusted by a factor of 0.1 after every epoch.
 
 * 3.2.4. Training:
     
    | Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate Reductions |
    | --- | --- | --- | --- | --- |  --- |
    | 1 | 0.2169 | 0.9145 | 0.3930 | 0.8612 | Manual LR reduction to 0.0001 (1e-4) |
    | 2 | 0.0529 | 0.9832 | 0.1947 | 0.9361 | Manual LR reduction to 0.00001 (1e-5) |
    | **3** | 0.0307 | 0.9917 | **0.1871** | **0.9408** | Manual LR reduction to 0.000001 (1e-6) |
    | 4 | 0.0280 | 0.9925 | 0.1856 | 0.9401 | Manual LR reduction to 0.0000001 (1e-7) |
    | 5 | 0.0278 | 0.9927 | 0.1896 | 0.9390 | Manual LR reduction to 0.00000001 (1e-8) |
    | 6 | 0.0277 | 0.9928 | 0.1902 | 0.9398 | None |

![loss](https://github.com/Ma-Marcinowski/Verificational-Model/blob/master/Plots/VM_v2.1.0_Loss.png "Model Loss") ![acc](https://github.com/Ma-Marcinowski/Verificational-Model/blob/master/Plots/VM_v2.1.0_Accuracy.png "Model Accuracy")

* #### 3.3. Model v2.1.0 evaluation on [256x256] patches
   
  * 3.3.1. Database:
  
    * Vide 2.3.1. Database.
  
  * 3.3.2. Metrics:
  
    * Vide 2.3.2. Metrics.
  
  * 3.3.3. CVL evaluation:
    
    | Criterion | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | None | 3 | 0.1835 | 0.9387 | 0.9301 | 0.9474 | 0.0526 | 0.0699 | 0.9464 | 0.9312 | 0.9817 |
    | **Hard** | 3 | **0.1494** | **0.9506** | 0.9567 | 0.9445 | 0.0555 | 0.0433 | 0.9452 | 0.9561 | 0.9858 |
  
    * Epochs of model training - EofT - by the best validation accuracy result; 
  
  * 3.3.4. IAM evaluation:
  
    | Denoised | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | **False** | 3 | 283.7893 | 0.5266 | **0.9883** | **0.0649** | 0.9350 | 0.0116 | 0.5138 | 0.8476 | 0.5269 |
    | True | 3 | 0.6610 | 0.9138 | 0.9867 | 0.8408 | 0.1592 | 0.0133 | 0.8611 | 0.9845 | 0.9510 |
   
    * Epochs of model training - EofT - by the best validation accuracy result;   
   
* #### 3.4. Model v2.1.1 training on [256x256] patches (simplified variant of model v2.1.0)

  * 3.4.1. Simplified Model:
    
    * Exactly the same as model v2.1.0, except for additional output neuron and distance layers.

  * 3.4.2. Database:
  
    * Vide 2.2.1. Database.
 
  * 3.4.3. Callbacks: 
  
    * Vide 2.2.2. Callbacks.
  
  * 3.4.4. Hyperparameters: 
    
    * Vide 2.2.3. Hyperparameters;
    * Epochs - 3 (three separate runs);
    * Learning rate - initial 0.001 (1e-3), then manually adjusted by a factor of 0.1 after every epoch.
 
  * 3.4.5. Training:
     
    | Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate Reductions |
    | --- | --- | --- | --- | --- |  --- |
    | 1 | 0.4259 | 0.8096 | 0.3368 | 0.8561 | Manual LR reduction to 0.0001 (1e-4) |
    | **2** | 0.2462 | 0.9095 | **0.2834** | **0.8866** | Manual LR reduction to 0.00001 (1e-5) |
    | 3 | 0.2069 | 0.9266 | 0.2926 | 0.8831 | None |
    
* #### 3.5. Model v2.2.0 training on [256x256] patches (binarized images)

  * 3.5.1. Database:
  
    * Preprocessing v0.2
  
    * Training dataset - CVL database subset of 1415 document images by 283 writers - 991448 image pairs (equal number of positive and negative instances);
        
    * Validation dataset - CVL database subset of 189 document images by 27 writers - 20% of test instances.

  * 3.5.2. Hyperparameters: 
    
    * Vide 2.2.3. Hyperparameters;
    * Epochs - 5 (five separate runs);
    * Learning rate - initial 0.001 (1e-3), then manually adjusted by a factor of 0.1 after every epoch.
 
  * 3.5.3. Training:
     
    | Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate Reductions |
    | --- | --- | --- | --- | --- | --- |
    | 1 | 0.2208 | 0.9091 | 0.3238 | 0.9057 | Manual LR reduction to 0.0001 (1e-4) |
    | 2 | 0.0440 | 0.9862 | 0.2989 | 0.9245 | Manual LR reduction to 0.00001 (1e-5) |
    | 3 | 0.0220 | 0.9944 | 0.2755 | 0.9276 | Manual LR reduction to 0.00001 (1e-6) |
    | **4** | 0.0195 | 0.9952 | **0.2728** | **0.9284** | Manual LR reduction to 0.00001 (1e-7) |
    | 5 | 0.0192 | 0.9953 | 0.2746 | 0.9280 | None |

* #### 3.6. Model v2.2.0 evaluation on [256x256] patches (binarized images)

  * 3.6.1. Database:
  
    * Preprocessing v0.2
    
    * Test dataset - CVL database subset of 189 document images by 27 writers - 156600 image pairs (equal number of positive and negative instances).
  
  * 3.6.2. Metrics:
  
    * Vide 2.3.2. Metrics.
  
  * 3.6.3. CVL evaluation:
  
    | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | 4 | 0.2783 | 0.9275 | 0.8941 | 0.9608 | 0.0392 | 0.1059 | 0.9580 | 0.9007 | 0.9749 |
   
    * Epochs of model training - EofT - by the best validation accuracy and loss result;   
    
* #### 3.7. Model v2.3.0 training on [256x256] patches (extended train database of binarized images)

  * 3.7.1. Model v2.3.0:
  
    * Exactly the same as v2.1.0, except for training on combained IAM and CVL databases (of binarized images).
     
  * 3.7.2. Database:
  
    * Preprocessing v0.3 (binarized images, only IAM images are denoised);
      
    * Training dataset - a subset of combined CVL and IAM databases, containing 2740 document images (1415 from CVL and 1325 from IAM) by 822 writers (283 from CVL and 539 from IAM) - 3493614 image pairs (equal number of positive and negative instances). Dataframe split into 2 equal size parts;
        
    * Validation dataset - a subset of combined CVL and IAM databases, containing 403 document images (189 from CVL and 214 from IAM) by 145 writers (27 from CVL and 118 from IAM) - 20% of test instances.
    
  * 3.7.3. Hyperparameters: 
    
    * Vide 2.2.3. Hyperparameters;
    * Epochs - one epoch per one training dataframe part (each ran separately);
    * Learning rate - initial 0.001 (1e-3).
 
  * 3.7.4. Training (learning rate reductions by a factor of 0.1):
                                                                                                                                                                                                                
    | Epoch | TDP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate Reductions |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/2 | 0.1309 | 0.9497 | 0.1714 | 0.9480 | Manual LR reduction to 0.0001 (1e-4) |
    | 2 | 2/2 | 0.0510 | 0.9826 | 0.1820 | 0.9482 | Manual LR reduction to 0.00001 (1e-5) |
    | 3 | 1/2 | 0.0297 | 0.9907 | 0.1808 | 0.9518 | Manual LR reduction to 0.000001 (1e-6) |
    | 4 | 2/2 | 0.0304 | 0.9905 | 0.1809 | 0.9520 | None |
    
    * Training dataframe part - TDP - utilized for a given epoch of training is indicated by its index;
 
  * 3.7.5. Training rerun of epochs following the first one (learning rate reductions by a factor of 0.01):
    
    | Epoch | TDP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate Reductions |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/2 | 0.1309 | 0.9497 | 0.1714 | 0.9480 | Manual LR reduction to 0.00001 (1e-5) |
    | 2 | 2/2 | 0.0612 | 0.9789 | 0.1541 | 0.9543 | Manual LR reduction to 0.0000001 (1e-7) |
    | 3 | 1/2 | 0.0430 | 0.9863 | 0.1530 | 0.9547 | Manual LR reduction to 0.000000001 (1e-9) |
    | **4** | 2/2 | 0.0559 | 0.9808 | **0.1513** | **0.9552** | None |
    | 5 | 1/2 | 0.0429 | 0.9863 | 0.1517 | 0.9552 | None |
    
    * Training dataframe part - TDP - utilized for a given epoch of training is indicated by its index;  
      
* #### 3.8. Model v2.3.0 evaluation on [256x256] patches (extended train database of binarized images)

  * 3.8.1. Database:
  
    * Test dataset - a subset of combined CVL and IAM databases, containing 403 document images (189 from CVL and 214 from IAM) by 145 writers (27 from CVL and 118 from IAM) - 406548 image pairs (equal number of positive and negative instances);
        
    * CVL criterion - a test subset of CVL database - 156600 image pairs (equal number of positive and negative instances);
    
    * IAM criterion - a test subset of IAM database - 228172 image pairs (equal number of positive and negative instances);
    
    * Hard criterion - excluded documents containing the same samlpe text as train documents (*ergo* included documents containing only samlpe texts no. 7 and 8 in the case of CVL database) - 30496 image pairs (equal number of positive and negative instances). IAM test subset is omitted, because during the standard and IAM criterion test already no IAM test documents did contain the same samlpe text as IAM train documents;
    
    * Negative criterion - a subset of combined CVL and IAM databases, containg only negative instances of cross databases image pairs, such that for any given `xy` pair, an `x` belongs to CVL testset and `y` to IAM testset - 200000 image pairs (an arbitrary number);
    
    * Average criterion - metrics averaged over separate CVL and IAM tests.
    
  * 3.8.2. Metrics:

    * Vide 2.3.2. Metrics.

  * 3.8.3. CVL and IAM evaluation:
  
    | Criterion | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | None | 4 | 0.1477 | 0.9566 | 0.9523 | 0.9608 | 0.0392 | 0.0477 | 0.9605 | 0.9527 | 0.9878 |
    | IAM | 4 | 0.1816 | 0.9425 | 0.9488 | 0.9363 | 0.0637 | 0.0512 | 0.9371 | 0.9481 | 0.9824 |
    | CVL | 4 | 0.2180 | 0.9284 | 0.9583 | 0.8985 | 0.1015 | 0.0417 | 0.9042 | 0.9556 | 0.9772 |
    | Hard | 4 | 0.2189 | 0.9290 | 0.9616 | 0.8964 | 0.1036 | 0.0384 | 0.9027 | 0.9589 | 0.9789 |
    | **Negative** | 4 | **0.0017** | **0.9998** | None | 0.9998 | 0.0002 | None | None | None | None |
    | **Average** | 4 | **0.1998** | **0.9354** | 0.9535 | 0.9174 | 0.0826 | 0.0464 | 0.9206 | 0.9518 | 0.9798 |
       
    * Epochs of model training - EofT - by the best validation accuracy and loss result;   
    
* #### 3.9. Model v2.4.0 training on [256x256] patches (extended train database of binarized and denoised images)

  * 3.9.1. Model v2.4.0:
  
    * Exactly the same as v2.1.0, except for training on extended database of binarized and denoised images.
  
  * 3.9.2. Database:
  
    * Vide 3.7.2. Database, except for denoiseing of all images (preprocessing v0.4).
    
  * 3.9.3. Hyperparameters:
  
    * Vide 3.7.3. Hyperparameters.
    
  * 3.9.4 Training (learning rate reductions by a factor of 0.01):
  
    | Epoch | TDP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate Reductions |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/2 | 0.1581 | 0.9365 | 0.2879 | 0.9028 | Manual LR reduction to 0.00001 (1e-5) |
    | 2 | 2/2 | 0.0701 | 0.9750 | 0.1649 | 0.9484 | Manual LR reduction to 0.0000001 (1e-7) |
    | 3 | 1/2 | 0.0502 | 0.9835 | 0.1637 | 0.9480 | Manual LR reduction to 0.000000001 (1e-9) |
    | 4 | 2/2 | 0.0642 | 0.9774 | 0.1622 | 0.9489 | Manual LR reduction to 0.00000000001 (1e-11) |
    | **5** | 1/2 | 0.0502 | 0.9834 | **0.1616** | **0.9489** | None |
    
    * Training dataframe part - TDP - utilized for a given epoch of training is indicated by its index;  
  
* #### 3.10. Model v2.4.0 evaluation on [256x256] patches (extended train database of binarized and denoised images)

  * 3.10.1. Database:
  
    * Vide 3.8.1. Database, except for preprocessing v0.4;
  
  * 3.10.2. Metrics:
  
    * Vide 2.3.2. Metrics.
  
  * 3.10.3. CVL and IAM evaluation:
  
    | Criterion | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | None | 5 | 0.1661 | 0.9481 | 0.9330 | 0.9632 | 0.0368 | 0.0670 | 0.9621 | 0.9349 | 0.9859 |
    | IAM | 5 | 0.1839 | 0.9420 | 0.9323 | 0.9517 | 0.0483 | 0.0677 | 0.9507 | 0.9336 | 0.9833 |
    | CVL | 5 | 0.2446 | 0.9153 | 0.9336 | 0.8969 | 0.1031 | 0.0664 | 0.9006 | 0.9311 | 0.9689 |
    | Hard | 5 | 0.2429 | 0.9207 | 0.9389 | 0.9025 | 0.0975 | 0.0611 | 0.9060 | 0.9366 | 0.9706 |
    | **Negative** | 5 | **0.0209** | **0.9922** | None | 0.9922 | 0.0078 | None | None | None | None |
    | **Average** | 5 | **0.2142** | **0.9286** | 0.9329 | 0.9243 | 0.0757 | 0.0670 | 0.9256 | 0.9323 | 0.9761 |
       
    * Epochs of model training - EofT - by the best validation accuracy and loss result;   

* #### 3.11. Model v2.5.0 training on [256x256] patches (extended train database of grayscaled and noised images)

  * 3.11.1. Model v2.5.0:
  
    * Exactly the same as v2.1.0, except for training on a database generated via preprocessing v0.5.
  
  * 3.11.2. Database:
    
    * Preprocessing v0.5;
      
    * Training dataset - a subset of combined CVL and IAM databases, containing 2740 document images (1415 from CVL and 1325 from IAM) by 822 writers (283 from CVL and 539 from IAM) - 2928660 image pairs (equal number of positive and negative instances);
        
    * Validation dataset - a subset of combined CVL and IAM databases, containing 403 document images (189 from CVL and 214 from IAM) by 145 writers (27 from CVL and 118 from IAM) - 20% of test instances.
    
  * 3.11.3. Hyperparameters:
  
    * Vide 3.7.3. Hyperparameters.
  
  * 3.11.4 Training (learning rate reductions by a factor of 0.01):
  
    | Epoch | TDP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate Reductions |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/2 | 0.1580 | 0.9407 | 0.2820 | 0.9097 | Manual LR reduction to 0.00001 (1e-5) |
    | 2 | 2/2 | 0.0515 | 0.9829 | 0.1484 | 0.9576 | Manual LR reduction to 0.0000001 (1e-7) |
    | 3 | 1/2 | 0.0458 | 0.9851 | 0.1468 | 0.9582 | Manual LR reduction to 0.000000001 (1e-9) |
    | 4 | 2/2 | 0.0607 | 0.9783 | 0.1467 | 0.9583 | Manual LR reduction to 0.00000000001 (1e-11) |
    | **5** | 1/2 | 0.0457 | 0.9849 | **0.1466** | **0.9584** | None |
    
    * Training dataframe part - TDP - utilized for a given epoch of training is indicated by its index;  
 
* #### 3.12. Model v2.5.0 evaluation on [256x256] patches (extended train database of grayscaled and noised images)

  * 3.12.1. Database:
    
    * Vide 3.10.1. Database, except for preprocessing v0.5;
  
    * Test dataset - 325604 image pairs (equal number of positive and negative instances);
    
    * CVL criterion - 143092 image pairs (equal number of positive and negative instances);
    
    * IAM criterion - 182513 image pairs (equal number of positive and negative instances);
    
    * Hard criterion - 13234 image pairs (equal number of positive and negative instances);
    
    * Negative criterion - 200000 image pairs (an arbitrary number);
    
    * Average criterion - metrics averaged over separate CVL and IAM tests.
    
  * 3.12.2. Metrics:
  
    * Vide 2.3.2. Metrics.
  
  * 3.12.3. CVL and IAM evaluation:
  
    | Criterion | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | **None** | 5 | 0.1463 | **0.9589** | 0.9576 | 0.9602 | 0.0398 | 0.0424 | 0.9601 | 0.9577 | 0.9879 |
    | IAM | 5 | 0.1261 | 0.9653 | 0.9608 | 0.9697 | 0.0303 | 0.0392 | 0.9694 | 0.9611 | 0.9906 |
    | CVL | 5 | 0.1819 | 0.9459 | 0.9535 | 0.9382 | 0.0618 | 0.0465 | 0.9392 | 0.9528 | 0.9832 |
    | Hard | 5 | 0.1415 | 0.9560 | 0.9696 | 0.9424 | 0.0576 | 0.0304 | 0.9439 | 0.9688 | 0.9881 |
    | **Negative** | 5 | 0.0476| **0.9815** | None | 0.9815 | 0.0185 | None | None | None | None |
    | **Average** | 5 | 0.1540 | **0.9556** | 0.9571 | 0.9539 | 0.0460 | 0.0428 | 0.9543 | 0.9569 | 0.9869 |
       
    * Epochs of model training - EofT - by the best validation accuracy and loss result;
    
* #### 3.13. Model v2.5.1 training on [256x256] patches (extended train database of raw grayscaled images)

  * 3.13.1. Model v2.5.1:
  
    * Exactly the same as v2.5.0, except for training on images neither denoised nor noised.
  
  * 3.13.2. Database:
    
    * Preprocessing v0.5, except for denoising and addition of noise to images (ergo images are simply extracted, grayscaled, inverted, resized and cropped into patches);
      
    * Training dataset - a subset of combined CVL and IAM databases, containing 2740 document images (1415 from CVL and 1325 from IAM) by 822 writers (283 from CVL and 539 from IAM) - 2928660 image pairs (equal number of positive and negative instances);
        
    * Validation dataset - a subset of combined CVL and IAM databases, containing 403 document images (189 from CVL and 214 from IAM) by 145 writers (27 from CVL and 118 from IAM) - 20% of test instances.
    
  * 3.13.3. Hyperparameters:
  
    * Vide 3.7.3. Hyperparameters.
  
  * 3.13.4 Training (learning rate reductions by a factor of 0.01):
  
    | Epoch | TDP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate Reductions |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/2 | 0.1603 | 0.9419 | 0.3005 | 0.8976 | Manual LR reduction to 0.00001 (1e-5) |
    | 2 | 2/2 | 0.0546 | 0.9808 | 0.1326 | 0.9583 | Manual LR reduction to 0.0000001 (1e-7) |
    | 3 | 1/2 | 0.0385 | 0.9876 | 0.1319 | 0.9584 | Manual LR reduction to 0.000000001 (1e-9) |
    | 4 | 2/2 | 0.0492 | 0.9828 | 0.1318 | 0.9579 | Manual LR reduction to 0.00000000001 (1e-11) |
    | 5 | 1/2 | 0.0384 | 0.9875 | 0.1319 | 0.9587 | None |
    | 6 | 2/2 | 0. | 0. | 0. | 0. | None |

    
    * Training dataframe part - TDP - utilized for a given epoch of training is indicated by its index;  
 
* #### 3.14. Model v2.5.1 evaluation on [256x256] patches (extended train database of raw grayscaled images)

  * 3.14.1. Database:
    
    * Vide 3.10.1. Database, except for preprocessing v0.5;
  
    * Test dataset - 328824 image pairs (equal number of positive and negative instances);
    
    * CVL criterion - 144756 image pairs (equal number of positive and negative instances);
    
    * IAM criterion - 184068 image pairs (equal number of positive and negative instances);
    
    * Hard criterion - 13304 image pairs (equal number of positive and negative instances);
    
    * Negative Raw criterion - 200000 image pairs (an arbitrary number);
    
    * Negative Denoised criterion - 200000 denoised (by thresholding of pixel values lower than 55 to zero) image pairs (an arbitrary number);
    
    * Average criterion - metrics averaged over separate CVL and IAM tests.
    
  * 3.14.2. Metrics:
  
    * Vide 2.3.2. Metrics.
  
  * 3.14.3. CVL and IAM evaluation:
  
    | Criterion | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | None | 0 | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
    | IAM | 0 | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
    | CVL | 0 | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
    | Hard | 0 | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
    | Negative Raw | 0 | 0.| 0. | None | 0. | 0. | None | None | None | None |
    | Negative Denoised | 0 | 0.| 0. | None | 0. | 0. | None | None | None | None |
    | Average | 0 | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
       
    * Epochs of model training - EofT - by the best validation accuracy and loss result;   
    
* #### 3.15. Model v2.5.1 evaluation on [1024x1024] extracts (extended train database of raw grayscaled images)

  * 3.15.1. Database:
    
    * Vide 3.14.1. Database, except for evaluation on [1024x1024] document extracts.
  
    * Test dataset - ??? image pairs (equal number of positive and negative instances);
    
    * CVL criterion - ??? image pairs (equal number of positive and negative instances);
    
    * IAM criterion - ??? image pairs (equal number of positive and negative instances);
    
    * Hard criterion - ??? image pairs (equal number of positive and negative instances);
    
    * Negative Raw criterion - 200000 image pairs (an arbitrary number);
    
    * Negative Denoised criterion - 200000 image pairs (an arbitrary number);
    
    * Average criterion - metrics averaged over separate CVL and IAM tests.
    
  * 3.15.2. Metrics:
  
    * Vide 2.3.2. Metrics.
  
  * 3.15.3. CVL and IAM evaluation:
  
    | Criterion | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | None | 0 | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
    | IAM | 0 | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
    | CVL | 0 | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
    | Hard | 0 | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
    | Negative Raw | 0 | 0.| 0. | None | 0. | 0. | None | None | None | None |
    | Negative Denoised | 0 | 0.| 0. | None | 0. | 0. | None | None | None | None |
    | Average | 0 | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
       
    * Epochs of model training - EofT - by the best validation accuracy and loss result;   

* #### 3.16. Model 2.6.0 combined evaluation on [256x256] patches (extended train database of ??? images)

  * 3.16.1. Method of combined evaluation:     

    * For a given pair of documents `XY`, the prediction is based on a combination of all possible partial predictions, i.e. predictions of such `xy` patch pairs, that `x` is a patch of an `X` document and `y` is a patch of an `Y` document;
    
    * Hence all positive and all negative pairs need to be generated (except for pair reverses);
    
    * A given partial prediction is positive if the result is higher than 0.75, and negative if the result is lower than 0.25, else patch prediction is unresolved;
    
    * A given combined prediction is positive if the number of all partial positive predictions is higher than a sum of negative and unresolved partial predictions. Combined negative or unresolved predictions are analogously defined.
  
  * 3.16.2. Model:
  
    * Model v2.5.? (neither retrained, nor retested via preprocessing v0.6);
    
    * Model training, vide 3.??.3.;
    
    * Model evaluation, vide 3.??.4.;
    
    * However another test dataframe is generated by a method of preprocessing v0.6, for the purpose of combined evaluation. 

  * 3.16.3. Test dataset:
  
      * Preprocessing v0.6;
  
      * A subset of combined CVL and IAM databases, containing 403 document images (189 from CVL and 214 from IAM) by 145 writers (27 from CVL and 118 from IAM); 
      
      * 2935782 image pairs (2852699 negative and 83083 positive instances). 0
      * 2970832 image pairs (2886934 negative and 83898 positive instances). 1
  
  * 3.16.4. Metrics:
  
    * Vide 2.3.2. Metrics, except for Loss and AUC;
    
    * Rate of unresolved instances - `(Number unresolved instances / total number of instances)` - UNR. 
  
  * 3.16.5. CVL and IAM cumulative evaluation of model v2.5.?:
  
    | EofT | Acc | TPR | TNR | FPR | FNR | PPV | NPV | UNR |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | 0 | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
   
    * Epochs of model training - EofT - by the best validation accuracy and loss result;
    
* #### 3.17. Model 2.7.0 training on [256x256] patches (extended train database of ??? images)
* #### 3.18. Model 2.7.0 evaluation on [256x256] patches (extended train database of ??? images)
* #### 3.19. Model 2.7.0 combined evaluation on [256x256] patches (extended train database of ??? images)    

### 4. Verificational Model v3
