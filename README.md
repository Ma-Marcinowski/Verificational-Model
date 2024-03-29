## Verificational Model

### 0. Introduction

* #### 0.1. Objective of the author's repository was to introduce multiple varying methods for verification of offline handwritten documents authorship by artificial neural networks (through classification of preprocessed image pairs to positive `same author` or negative `different authors` class). The purpose of such methods, models and experiments as described below, was to create an empirical background for scientific analysis of machine learning tools developed in the field of computational forensics.

* #### 0.2. Author's original assumptions were that:
     
    * Document images will be best processed simultaneously by two separate convolutional neural networks and classified by one multilayer perceptron (fully connected layers);       
    
    * Preprocessing shouldn't drastically affect image quality (e.g. by image binarization) to preserve most of handwriting features.
    
    * The number of positive and negative class instances (generated for the purpose of model training / testing) ought to be equal.

* #### 0.3. Versioning

   * 0.3.1 Experiment identifiers (e.g. vX.Y.Z):

      * X indicates the model version;
      * Y indicates the method of preprocessing;
      * Z indicates any extra variation of a given X.Y base combination.

   * 0.3.2 Model version identifiers (e.g. vX):

      * X indicates the model version (as already stated in 0.3.1).

   * 0.3.3 Preprocessing method identifiers (e.g. v0.Y):
    
      * X is zeroed and Z skipped to avoid any confusion of model versions and preprocessing methods; 
      * Y indicates the method of preprocessing (as already stated in 0.3.1).

* #### 0.4. Keywords:

    * Computational, statistical, probabilistic; 
    
    * Forensics, criminalistics, analysys, examination;
    
    * Handwriting, signatures, documents;
    
    * Neural, networks, deep, machine, learning, artificial, intelligence, ANN, AI;

    * Methods, criteria, categories, evaluation, falsification, veryfication. 

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
            
    * Step two `Dataframe_v0.1.py` - creation of a dataframe (i.e. of a `csv` file that can be edited in any spreadsheet program, e.g. calc / excel) separately for the test and training subsets, by combinatorial pairing of image names into the positive class, and a random combinatorial pairing of image names into the negative class (the number of possible negative combinations is much greater than the number positive ones, so all positive instances are created first and then the negative instances are randomly combinated until their number is equal to the number of positive instances). It has to be noted however, that for any given positive `xy` pair, also the reverse pair `yx` will be created (in the case of negative class, there is a possibility that for any randomly generated `xy` also `yx` will be randomly generated). Image name pairs and their labels are ordered by rows, according to columns `left convolutional path`, `right convolutional path`, and `label` (labels are determined by the convergence or divergence of authors' identifiers - e.g. first four digits of a raw image name in the case of a CVL database). Above method requires that the test and training images are kept in different directories during their preprocessing. However, it is not necessary to create manually any of dataframe `csv` files, i.e. they will be created by the program (and if any such a file was already created manually, its directory and name may be indicated in the program code). Validation dataframe (utilized only for testing of the model during its training, generally after every epoch) is also created, by random sampling of the test dataframe instances (fratcion of which to pull has to be specified - usually 0.1 / 0.2 is enough for validation purposes). Due to the randomness of sampling, it is most probable that the number of sampled positive and negative instances will be effectively equal.
               
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
     
     * To minimize overfitting and perturbations (due to the noise present in some cases of IAM images), threshold of the function thresholding to zero is lowered to 15, and slight noise is added to all images (both CVL and IAM);
     
     * Because thresholding of empty patches by a mean of pixel values proved ineffective over time, another more subtle method is applied, i.e. any given patch is multiplied by a filter matrix (vide `/Examples/Preprocessing-Filters/` folder) and accepted if the sum of all it's elements is higher than a zero. It has to be noted that ultimately five filters are applied, therefore any given patch has to subsequently pass through five thresholds;
     
     * No cross-databases pairs are generated, i.e. negative instances of inter databases image pairs, such that for any given `xy` pair, an `x` belongs to CVL testset and `y` to IAM testset (or vice versa).     
     
### 1.6. Preprocessing v0.6 (CVL and IAM database, grayscaled and raw images)

* #### 1.6.1. Exactly the same as v0.5, however no noise is added or removed, and dataframes are generated differently:

    * No reverse pairs are created (neither positive nor negative), e.g. if a pair `xy` was already generated, then a pair `yx` is omitted;
    
    * Train and validation dataframes are nonetheless created under the assumption that the number of positive and negative instances ought to be equal; 
    
    * For the purpose of standard model testing, a validation dataframe is generated to the extent of possible positive instances, where the number of positive and negative instances is equal, and divided into `n` smaller validation dataframes utilized for the purpuose of rough testing between epochs;
    
    * The test dataframe is generated for the purpose of *combined evaluation* (vide 3.17.1. Method of combined evaluation), hence all possible negative and positive instances are created (except for pair reverses).
    
### 1.7. Preprocessing v0.7 (CVL and IAM database, grayscaled and raw images)

* #### 1.7.1. Exactly the same as v0.6, however:

    * Neither the test (i.e. combined evaluation) nor train dataframes are created under the assumption that the number of positive and negative instances ought to be equal;
    
    * Hence for the purpose of training, a dataframe of all possible positive instances and of `n` times more negative instances is generated (except for pair reverses) and divided into `m` equal parts;
    
* #### 1.7.2. Therefore:

    * Validation dataframes are generated the same way as in the case of v0.6;
     
    * The test dataframe is generated for the purpose of *combined evaluation* exactly as in the case of v0.6.
    
### 1.8. Preprocessing v0.8 (CVL and IAM database, grayscaled and raw images)

* #### 1.8.1. Exactly the same as v0.6, however:

    * Left-channel images are [512 x 512] patches, and right-channel images are [256x256] patches.
  
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
       
    * Google Colaboratory - (2019) - Python 3 Jupyter Notebook, GPU type runtime (Nvidia Tesla K80), 290ms/step (27732 steps per epoch) in the case of model v1.1.0 training on [256x256] patches;
        
    * Google Kaggle - (2019) - Python 3 Jupyter Notebook, GPU type runtime (Nvidia Telsa P100), 65ms/step (27732 steps per epoch) in the case of model v1.1.0 training on [256x256] patches.
        
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
              
    * Epochs of model training - EofT - by the best validation loss result.
              
  * 2.3.4. IAM evaluation:
    
    | Denoised | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | **False** | 5 | 265.3726 | 0.5047 | **0.0113** | **0.9979** | 0.0020 | 0.9886 | 0.8478 | 0.5023 | 0.5058 |
    | True | 5 | 0.3629 | 0.8835 | 0.9661 | 0.8009 | 0.1990 | 0.0338 | 0.8291 | 0.9594 | 0.9603 |
    
    * Epochs of model training - EofT - by the best validation loss result.

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
       
    * Google Colaboratory - (2019) - Python 3 Jupyter Notebook, GPU type runtime (Nvidia Tesla K80), 250ms/step (27732 steps per epoch) in the case of model v2.1.0 training on [256x256] patches.    
  
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
  
    * Epochs of model training - EofT - by the best validation accuracy result. 
  
  * 3.3.4. IAM evaluation:
  
    | Denoised | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | **False** | 3 | 283.7893 | 0.5266 | **0.9883** | **0.0649** | 0.9350 | 0.0116 | 0.5138 | 0.8476 | 0.5269 |
    | True | 3 | 0.6610 | 0.9138 | 0.9867 | 0.8408 | 0.1592 | 0.0133 | 0.8611 | 0.9845 | 0.9510 |
   
    * Epochs of model training - EofT - by the best validation accuracy result.   
   
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
   
    * Epochs of model training - EofT - by the best validation accuracy and loss result.   
    
* #### 3.7. Model v2.3.0 training on [256x256] patches (extended train database of binarized images)

  * 3.7.1. Model v2.3.0:
  
    * Exactly the same as v2.1.0, except for training on combained IAM and CVL databases (of binarized images).
     
  * 3.7.2. Database:
  
    * Preprocessing v0.3 (binarized images, only IAM images are denoised);
      
    * Training dataset - a subset of combined CVL and IAM databases, containing 2740 document images (1415 from CVL and 1325 from IAM) by 822 writers (283 from CVL and 539 from IAM) - 3493614 image pairs (equal number of positive and negative instances) divided into two equal training dataframes;
        
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
    
    * Training dataframe part - TDP - utilized for a given epoch of training is indicated by its index.
 
  * 3.7.5. Training rerun of epochs following the first one (learning rate reductions by a factor of 0.01):
    
    | Epoch | TDP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate Reductions |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/2 | 0.1309 | 0.9497 | 0.1714 | 0.9480 | Manual LR reduction to 0.00001 (1e-5) |
    | 2 | 2/2 | 0.0612 | 0.9789 | 0.1541 | 0.9543 | Manual LR reduction to 0.0000001 (1e-7) |
    | 3 | 1/2 | 0.0430 | 0.9863 | 0.1530 | 0.9547 | Manual LR reduction to 0.000000001 (1e-9) |
    | **4** | 2/2 | 0.0559 | 0.9808 | **0.1513** | **0.9552** | None |
    | 5 | 1/2 | 0.0429 | 0.9863 | 0.1517 | 0.9552 | None |
    
    * Training dataframe part - TDP - utilized for a given epoch of training is indicated by its index.  
      
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
    | **None** | 4 | **0.1477** | **0.9566** | 0.9523 | 0.9608 | 0.0392 | 0.0477 | 0.9605 | 0.9527 | 0.9878 |
    | IAM | 4 | 0.1816 | 0.9425 | 0.9488 | 0.9363 | 0.0637 | 0.0512 | 0.9371 | 0.9481 | 0.9824 |
    | CVL | 4 | 0.2180 | 0.9284 | 0.9583 | 0.8985 | 0.1015 | 0.0417 | 0.9042 | 0.9556 | 0.9772 |
    | Hard | 4 | 0.2189 | 0.9290 | 0.9616 | 0.8964 | 0.1036 | 0.0384 | 0.9027 | 0.9589 | 0.9789 |
    | **Negative** | 4 | **0.0017** | **0.9998** | None | 0.9998 | 0.0002 | None | None | None | None |
    | **Average** | 4 | **0.1998** | **0.9354** | 0.9535 | 0.9174 | 0.0826 | 0.0464 | 0.9206 | 0.9518 | 0.9798 |
       
    * Epochs of model training - EofT - by the best validation accuracy and loss result.   
    
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
    
    * Training dataframe part - TDP - utilized for a given epoch of training is indicated by its index. 
  
* #### 3.10. Model v2.4.0 evaluation on [256x256] patches (extended train database of binarized and denoised images)

  * 3.10.1. Database:
  
    * Vide 3.8.1. Database, except for preprocessing v0.4;
  
  * 3.10.2. Metrics:
  
    * Vide 2.3.2. Metrics.
  
  * 3.10.3. CVL and IAM evaluation:
  
    | Criterion | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | **None** | 5 | **0.1661** | **0.9481** | 0.9330 | 0.9632 | 0.0368 | 0.0670 | 0.9621 | 0.9349 | 0.9859 |
    | IAM | 5 | 0.1839 | 0.9420 | 0.9323 | 0.9517 | 0.0483 | 0.0677 | 0.9507 | 0.9336 | 0.9833 |
    | CVL | 5 | 0.2446 | 0.9153 | 0.9336 | 0.8969 | 0.1031 | 0.0664 | 0.9006 | 0.9311 | 0.9689 |
    | Hard | 5 | 0.2429 | 0.9207 | 0.9389 | 0.9025 | 0.0975 | 0.0611 | 0.9060 | 0.9366 | 0.9706 |
    | **Negative** | 5 | **0.0209** | **0.9922** | None | 0.9922 | 0.0078 | None | None | None | None |
    | **Average** | 5 | **0.2142** | **0.9286** | 0.9329 | 0.9243 | 0.0757 | 0.0670 | 0.9256 | 0.9323 | 0.9761 |
       
    * Epochs of model training - EofT - by the best validation accuracy and loss result.  
    
* #### 3.11. Model v2.5.0 training on [256x256] patches (extended train database of grayscaled and noised images)

  * 3.11.1. Model v2.5.0:
  
    * Exactly the same as v2.1.0, except for training on a database generated via preprocessing v0.5.
  
  * 3.11.2. Database:
    
    * Preprocessing v0.5;
      
    * Training dataset - a subset of combined CVL and IAM databases, containing 2740 document images (1415 from CVL and 1325 from IAM) by 822 writers (283 from CVL and 539 from IAM) - 2928660 image pairs (equal number of positive and negative instances) divided into two equal training dataframes;
        
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
    
    * Training dataframe part - TDP - utilized for a given epoch of training is indicated by its index.  
 
* #### 3.12. Model v2.5.0 evaluation on [256x256] patches (extended train database of grayscaled and noised images)

  * 3.12.1. Database:
    
    * Vide 3.10.1. Database, except for preprocessing v0.5;
  
    * Test dataset - 325604 image pairs (equal number of positive and negative instances);
    
    * CVL criterion - 143092 image pairs (equal number of positive and negative instances);
    
    * IAM criterion - 182513 image pairs (equal number of positive and negative instances);
    
    * Hard criterion - 13234 image pairs (equal number of positive and negative instances);
    
    * Negative criterion - 200000 image pairs (an arbitrary number);
    
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
       
    * Epochs of model training - EofT - by the best validation accuracy and loss result.
    
* #### 3.13. Model v2.5.1 training on [256x256] patches (extended train database of raw grayscaled images)

  * 3.13.1. Model v2.5.1:
  
    * Exactly the same as v2.5.0, except for training on images neither denoised nor noised.
  
  * 3.13.2. Database:
    
    * Preprocessing v0.5, except for denoising and addition of noise to images (ergo images are simply extracted, grayscaled, inverted, resized and cropped into patches);
      
    * Training dataset - a subset of combined CVL and IAM databases, containing 2740 document images (1415 from CVL and 1325 from IAM) by 822 writers (283 from CVL and 539 from IAM) - 2928660 image pairs (equal number of positive and negative instances) divided into two equal training dataframes;
        
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
    | **5** | 1/2 | 0.0384 | 0.9875 | **0.1319** | **0.9587** | Manual LR reduction to 0.0000000000001 (1e-13) |
    | 6 | 2/2 | 0.0493 | 0.9828 | 0.1351 | 0.9581 | None |

    * Training dataframe part - TDP - utilized for a given epoch of training is indicated by its index.
 
* #### 3.14. Model v2.5.1 evaluation on [256x256] patches (extended train database of raw grayscaled images)

  * 3.14.1. Database:
    
    * Vide 3.10.1. Database, except for preprocessing v0.5;
  
    * Test dataset - 328824 image pairs (equal number of positive and negative instances);
    
    * CVL criterion - 144756 image pairs (equal number of positive and negative instances);
    
    * IAM criterion - 184068 image pairs (equal number of positive and negative instances);
    
    * Hard criterion - 13304 image pairs (equal number of positive and negative instances);
    
    * Negative Raw criterion - 200000 image pairs (an arbitrary number);
    
    * Negative Denoised criterion - 200000 denoised (by thresholding of pixel values lower than 55 to zero) image pairs (an arbitrary number).
       
  * 3.14.2. Metrics:
  
    * Vide 2.3.2. Metrics.
  
  * 3.14.3. CVL and IAM evaluation:
  
    | Criterion | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | **None** | 5 | 0.1319 | **0.9582** | 0.9529 | 0.9636 | 0.0364 | 0.0471 | 0.9632 | 0.9534 | 0.9891 |
    | IAM | 5 | 0.1150 | 0.9661 | 0.9607 | 0.9714 | 0.0286 | 0.0393 | 0.9711 | 0.9611 | 0.9912 |
    | CVL | 5 | 0.1628 | 0.9445 | 0.9430 | 0.9459 | 0.0541 | 0.0570 | 0.9458 | 0.9431 | 0.9846 |
    | Hard | 5 | 0.1284 | 0.9557 | 0.9654 | 0.9459 | 0.0541 | 0.0346 | 0.9469 | 0.9647 | 0.9892 |
    | **Negative Raw** | 5 | 0.0264 | **0.9897** | None | 0.9897 | 0.0103 | None | None | None | None |
    | **Negative Denoised** | 5 | 0.0818 | **0.9736** | None | 0.9736 | 0.0264 | None | None | None | None |
       
    * Epochs of model training - EofT - by the best validation accuracy and loss result.   
    
  * 3.14.4. Equalization of databases through removal of spurious filters:
  
     * Vide `Verificational-Model/Distribution-Based/DistributionEqivalentFilters_v2.5.1.py`
  
     * The average features distributions are calculated separately for IAM and CVL testsets;
     
     * Those distributions are then compared to determine those features, which were highly active in the case of one, but not the other testset (in a sense of the average feature values differences equal to or greater than a threshold of 0.25);
     
     * Such features (i.e. convolutional filters) are then removed from the model (i.e 89 out of 1024 filters were zeroed);
     
     * And the model is tested again (vide 3.14.5. Equalized CVL and IAM evaluation).
    
  * 3.14.5. CVL and IAM evaluation equalized through removal of spurious filters:

    | Criterion | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | **None** | 5 | 0.1640 | **0.9436** | 0.9831 | 0.9041 | 0.0959 | 0.0169 | 0.9111 | 0.9817 | 0.9884 |
    | IAM | 5 | 0.1400 | 0.9530 | 0.9861 | 0.9198 | 0.0802 | 0.0139 | 0.9248 | 0.9851 | 0.9907 |
    | CVL | 5 | 0.2109 | 0.9248 | 0.9794 | 0.8703 | 0.1297 | 0.0206 | 0.8830 | 0.9769 | 0.9841 |
    | Hard | 5 | 0.1956 | 0.9292 | 0.9873 | 0.8712 | 0.1288 | 0.0127 | 0.8846 | 0.9856 | 0.9876 |
    | **Negative Raw** | 5 | 0.1002 | **0.9588** | None | 0.9588 | 0.0412 | None | None | None | None |
    | **Negative Denoised** | 5 | 0.1828 | **0.9428** | None | 0.9428 | 0.0572 | None | None | None | None |
       
    * Epochs of model training - EofT - by the best validation accuracy and loss result. 
    
  * 3.14.6. Categorized:
  
     * Vide ` Verificational-Model/Categories/CVL_Test_Categorized.csv`;
     
     * Vide ` Verificational-Model/Categories/IAM_Test_Categorized.csv`;
     
  * 3.14.7. Categorized database:
  
     * None category - 145 authors, 328824 image pairs (equal number of positive and negative instances);
     
     * Feminine category - 66 authors, 76754 image pairs (equal number of positive and negative instances);
     
     * Masculine category - 79 authors, 91042 image pairs (equal number of positive and negative instances);
     
     * Sinistral category - 23 authors, 31406 image pairs (equal number of positive and negative instances);
     
     * Dextral category - 122 authors, 136390 image pairs (equal number of positive and negative instances);
     
     * Feminine Sinistral category - 9 authors, 10094 image pairs (equal number of positive and negative instances);
     
     * Masculine Sinistral category - 14 authors, 21312 image pairs (equal number of positive and negative instances);
     
     * Feminine Dextral category - 57 authors, 66660 image pairs (equal number of positive and negative instances);
     
     * Masculine Dextral category - 65 authors, 69730 image pairs (equal number of positive and negative instances);
       
 * 3.14.8. Probability of the given Acc results occuring (model v2.5.1 example)
  
     * Vide ` Verificational-Model/Evaluation/VM_Given_Acc_Probability_v.2.5.1.py`;
  
     * To estimate the quality of the following - and any - evaluation category or crieteria, it may be usefull to calculate the probability of randomly observing the results of interest, given some sample size (determined by the given category or criterion), and some arbitrary range of these results;
     
     * Here, the author calculated probabilities of observing the accuracy results, which were achieved during the following categorized evaluation (vide 3.14.9. evaluation);
     
     * The range of the expected results was set to plus-minus 0.50 of the observed results;
     
     * The limit of combinations (without repetition, where `n` is equal to the number of authors, and `k` is the authors sample) was set to a 1000 random combinations, hence, it is just an estimation of the given probability;
     
     * To allow for some background, all 1000 Acc results (depend on the limit of combinations) are turned into percents and rounded to intigers, then their dominant value is found, and the probability of such most observed Acc randomly occuring is calculated;
            
  * 3.14.9. Model v2.5.1 categorized evaluation on [256x256] patches (extended train database of raw grayscaled images)

    | PCARO | PDARO | Authors sample | Category | Loss | Category Acc | Dominant Acc | TPR | TNR | FPR | FNR | Category PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | **1.0000** | **1.0000** | **145** | **None** | **0.1319** | **0.9582** | **0.9600** | **0.9529** | **0.9636** | 0.0364 | 0.0471 | 0.9632 | 0.9534 | 0.9891 |
    | 0.2380 | 0.3440 | 66 | Feminine | 0.1672 | 0.9487 | 0.9600 | 0.9414 | 0.9560 | 0.0440 | 0.0586 | 0.9553 | 0.9422 | 0.9843 |
    | 0.3430 | 0.4030 | 79 | Masculine | 0.1094 | 0.9634 | 0.9600 | 0.9655 | 0.9614 | 0.0386 | 0.0345 | 0.9616 | 0.9653 | 0.9918 |
    | 0.1050 | 0.1780 | 23 | Sinistral | 0.1894 | 0.9383 | 0.9700 | 0.9360 | 0.9405 | 0.0595 | 0.0640 | 0.9403 | 0.9363 | 0.9807 |
    | 0.5530 | 0.5360 | 122 | Dextral | 0.1267 | 0.9592 | 0.9600 | 0.9587 | 0.9597 | 0.0403 | 0.0413 | 0.9597 | 0.9587 | 0.9895 |
    | **0.0392** | **0.2062** | **9** | **Feminine Sinistral** | **0.3308** | **0.9151** | **0.9800** | **0.8419** | **0.9883** | 0.0117 | 0.1581 | 0.9863 | 0.8621 | 0.9717 |
    | **0.0510** | **0.1920** | **14** | **Masculine Sinistral** | **0.1289** | **0.9485** | **0.9800** | **0.9806** | **0.9165** | 0.0835 | 0.0194 | 0.9215 | 0.9792 | 0.9911 | 
    | 0.2172 | 0.3120 | 57 | Feminine Dextral | 0.1517 | 0.9495 | 0.9600 | 0.9565 | 0.9426 | 0.0574 | 0.0435 | 0.9434 | 0.9559 | 0.9862 |
    | 0.3610 | 0.3420 | 65 | Masculine Dextral | 0.1272 | 0.9578 | 0.9600 | 0.9608 | 0.9547 | 0.0453 | 0.0392 | 0.9550 | 0.9606 | 0.9895 |
    
    * Probability of the given Category Acc Randomly Occuring - PCARO - given the authors sample size and the range of the expected Acc (±0.50).
    
    * Probability of the given Dominant Acc Randomly Occuring - PDARO - given the authors sample size and the range of the unexpected Acc (every result is multiplied by 100 and rounded to an intiger).
    
 * 3.14.10. Probability of all the Acc results occuring (model v2.5.1 example)
 
     * Vide `Verificational-Model/Evaluation/VM_Every_Acc_Probability_v.2.5.1.py`. 
 
     * Overall, it is more usefull - compared to 3.14.8 - to calculate the probability of random occuracne for all possible Acc results given some step, and all sample sizes given some step; 
     
     * The two dimensional visualisation:
     
          * Five authors' sample sizes were arbitrarily speicifed [2, 4, 8, 16, 32, 64, 128, 136, 144];

          * 1000 combinations - of authors - limit was set for every authors' sample size;

          * Per combination Acc results were multipied by 100 (turned into percents) and rounded to intigers, therefore there were 101 possible Acc values, i.e. the step was set to one.
          
     * The three dimensional visualisation:
          
          * Given 145 authors, the maximal 145 limit of authors' sample sizes was speicifed;
          
          * Sample sizes step was set to one;
          
          * The limit of author combinations was set to 500;
          
          * There were 101 possible Acc values (all Acc results were multipied by 100 and rounded to intigers).
     
     * Results of such probability approximations are discrete probability distributions (as opposed to the continuous PDFs).
     
![Probability_Distributions_VM_v2.5.1_2D](https://github.com/Ma-Marcinowski/Verificational-Model/blob/master/Plots/VM_v2.5.1_Acc_Probability_Distributions_2D.png "Model v2.5.1 Acc Probability Distributions 2D")

![Probability_Distributions_VM_v2.5.1_3D](https://github.com/Ma-Marcinowski/Verificational-Model/blob/master/Plots/VM_v2.5.1_Acc_Probability_Distributions_3D.png "Model v2.5.1 Acc Probability Distributions 3D")

* #### 3.15. Model v2.5.1 evaluation on [1024x1024] extracts (extended train database of raw grayscaled images)

  * 3.15.1. Database:
    
    * Vide 3.14.1. Database, except for evaluation on [1024x1024] document extracts.
  
    * Test dataset - 4478 image pairs (equal number of positive and negative instances);
    
    * CVL criterion - 2646 image pairs (equal number of positive and negative instances);
    
    * IAM criterion - 1832 image pairs (equal number of positive and negative instances);
    
    * Hard criterion - 216 image pairs (equal number of positive and negative instances);
    
    * Negative Raw criterion - 2000 image pairs (an arbitrary number);
    
    * Negative Denoised criterion - 2000 image pairs (an arbitrary number).
    
  * 3.15.2. Metrics:
  
    * Vide 2.3.2. Metrics.
  
  * 3.15.3. CVL and IAM evaluation:
  
    | Criterion | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | None | 5 | 0.1659 | 0.9352 | **0.9647** | **0.9058** | 0.0942 | 0.0353 | 0.9110 | 0.9625 | 0.9838 |
    | IAM | 5 | 0.1116 | 0.9580 | **0.9913** | **0.9247** | 0.0753 | 0.0087 | 0.9294 | 0.9906 | 0.9941 |
    | CVL | 5 | 0.1972 | 0.9233 | **0.9463** | **0.9002** | 0.0998 | 0.0537 | 0.9046 | 0.9437 | 0.9768 |
    | Hard | 5 | 0.2243 | 0.8935 | **0.9815** | **0.8056** | 0.1944 | 0.0185 | 0.8346 | 0.9775 | 0.9851 |
    | Negative Raw | 5 | 0.0226 | 0.9925 | None | 0.9925 | 0.0075 | None | None | None | None |
    | Negative Denoised | 5 | 0.0789 | 0.9735 | None | 0.9735 | 0.0265 | None | None | None | None |
       
    * Epochs of model training - EofT - by the best validation accuracy and loss result. 
    
  * 3.15.4. Equalization of results through features distribution equivalent subsets (threshold 0.05):
  
     * Vide `Verificational-Model/Distribution-Based/DistributionEqivalentImages_v2.5.1.py`
  
     * Distributions are calculated separately for every test extract and patch;
     
     * Those distributions are then normalized (L1-Norm) and compared (Cosine Distance) to determine pairs of extracts and patches, which are equivalent as to their features distribution and authorship;
     
     * Such equivalent patch and extract test subsets are created, in the case of which every patch has at least one equivalent – in terms of features distribution and authorship – extract;
     
     * Equivalence is defined as a cosine distance of feature distributions and the threshold of distance is set to a maximum of 0.05;
     
     * 70 pairs of extracts and patches were equivalent (24 extracts and 60 patches);
     
     * And the model is tested again (vide 3.15.5. Equalized CVL and IAM evaluation).
     
  * 3.15.5. Patches and Extracts evaluation equalized through features distribution equivalent subsets:

    | Size | Criterion | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | Patches | None | 0.0811 | 0.9716 | 0.9784 | 0.9648 | 0.0352 | 0.0216 | 0.9653 | 0.9781 | 0.9952 |
    | Extracts | None | 0.1174 | 0.9651 | 1.0000 | 0.9302 | 0.0698 | 0.0000 | 0.9348 | 1.0000 | 0.9884 |
    | Patches | IAM | 0.0516 | 0.9846 | 1.0000 | 0.9691 | 0.0309 | 0.0000 | 0.9700 | 1.0000 | 0.9961 |
    | Extracts | IAM | 0.0302 | 0.9857 | 1.0000 | 0.9714 | 0.0286 | 0.0000 | 0.9722 | 1.0000 | 1.0000 |
    | Patches | CVL | 0.0098 | 0.9924 | 1.0000 | 0.9848 | 0.0152 | 0.0000 | 0.9851 | 1.0000 | 1.0000 |
    | Extracts | CVL | 0.0324 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
    | Patches | Negative | 0.0054 | 0.9965 | None | 0.9965 | 0.0035 | None | None | None | None |
    | Extracts | Negative | 0.0436 | 0.9715 | None | 0.9715 | 0.0285 | None | None | None | None |
     
  * 3.15.6. Equalization of results through features distribution equivalent subsets (threshold 0.1):
  
     * Vide 3.15.4. Equalization of results through features distribution equivalent subsets (threshold 0.05).
     
     * However, the threshold of maximal distance is set to 0.1;
     
     * 2376 pairs of equivalent extracts and patches (159 extracts and 1035 patches);
    
  * 3.15.7. Patches and Extracts evaluation equalized through features distribution equivalent subsets:

    | Size | Criterion | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | Patches | None | 0.0710 | 0.9770 | 0.9961 | 0.9578 | 0.0422 | 0.0039 | 0.9594 | 0.9959 | 0.9965 |
    | Extracts | None | 0.1374 | 0.9481 | 0.9928 | 0.9034 | 0.0966 | 0.0072 | 0.9113 | 0.9920 | 0.9932 |
    | Patches | IAM | 0.0706 | 0.9766 | 0.9959 | 0.9573 | 0.0427 | 0.0041 | 0.9589 | 0.9957 | 0.9963 |
    | Extracts | IAM | 0.1157 | 0.9517 | 0.9909 | 0.9124 | 0.0876 | 0.0091 | 0.9188 | 0.9902 | 0.9963 |
    | Patches | CVL | 0.0850 | 0.9668 | 0.9971 | 0.9365 | 0.0635 | 0.0029 | 0.9401 | 0.9969 | 0.9976 |
    | Extracts | CVL | 0.2257 | 0.9036 | 1.0000 | 0.8072 | 0.1928 | 0.0000 | 0.8384 | 1.0000 | 0.9996 |
    | Patches | Hard | 0.0504 | 0.9764 | 1.0000 | 0.9529 | 0.0471 | 0.0000 | 0.9550 | 1.0000 | 0.9999 |
    | Extracts | Hard | 0.2150 | 0.8824 | 1.0000 | 0.7647 | 0.2353 | 0.0000 | 0.8095 | 1.0000 | 1.0000 |
    | Patches | Negative | 0.0292 |0.9888 | None | 0.9888 | 0.0112 | None | None | None | None |
    | Extracts | Negative | 0.0419 | 0.9859 | None | 0.9859 | 0.0141 | None | None | None | None |
    
* #### 3.16. Model v2.5.1 evaluation on [256x256] denoised and binarized patches (extended train database of raw grayscaled images)

  * 3.16.1. Database:
    
    * Vide 3.10.1. Database, except for preprocessing v0.5, but also binarization and denoising vide v.0.4;
  
    * Test dataset - 326922 image pairs (equal number of positive and negative instances);
    
    * CVL criterion - 144684 image pairs (equal number of positive and negative instances);
    
    * IAM criterion - 182238 image pairs (equal number of positive and negative instances);
    
    * Hard criterion - 12480 image pairs (equal number of positive and negative instances);
    
    * Negative criterion - 200000 image pairs (an arbitrary number).
       
  * 3.16.2. Metrics:
  
    * Vide 2.3.2. Metrics.
  
  * 3.16.3. CVL and IAM evaluation:
  
    | Criterion | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | **None** | 5 | 20.1986 | **0.5064** | **0.9971** | **0.0156** | **0.9844** | 0.0029 | 0.5032 | 0.8448 | 0.5111 |
    | IAM | 5 | 16.2168 | 0.5092 | 0.9950 | 0.0235 | 0.9765 | 0.0050 | 0.5047 | 0.8232 | 0.5160 |
    | CVL | 5 | 28.9528 | 0.5004 | 0.9999 | 0.0009 | 0.9991 | 0.0001 | 0.5002 | 0.8649 | 0.5007 |
    | Hard | 5 | 28.6659 | 0.5007 | 1.0000 | 0.0014 | 0.9986 | 0.0000 | 0.5004 | 1.0000 | 0.5013 |
    | **Negative** | 5 | 41.0964 | **0.0044** | **None** | **0.0044** | **0.9956** | None | None | None | None |
       
    * Epochs of model training - EofT - by the best validation accuracy and loss result.

* #### 3.16a. Model v2.5.1 evaluation on non-identical instances ([256x256] patches, extended train database of raw grayscaled images)

  * 3.16a.1. Database:
    
    * Vide 3.10.1. Database, except for preprocessing v0.5.

    * Test on non-identical patches:
  
         * Test dataset - 322056 image pairs (equal number of positive and negative instances);
    
         * CVL criterion - 142024 image pairs (equal number of positive and negative instances);
    
         * IAM criterion - 180032 image pairs (equal number of positive and negative instances);
    
         * Hard criterion - 12480 image pairs (equal number of positive and negative instances).
       
    * Test on non-identical patches from different documents:

         * Test dataset - 264832 image pairs (equal number of positive and negative instances);
   
         * CVL criterion - 122564 image pairs (equal number of positive and negative instances);
   
         * IAM criterion - 142268 image pairs (equal number of positive and negative instances);
    
         * Hard criterion - 6568 image pairs (equal number of positive and negative instances).
    
    * Test on non-identical patches from different documents and in different languages:

         * Test dataset / CVL criterion - 28538 image pairs (equal number of positive and negative instances);
    
         * Hard criterion  - 11826 image pairs (equal number of positive and negative instances).

         * Negative criterion - 200000 image pairs (an arbitrary number).

  * 3.16a.2. Metrics:
  
    * Vide 2.3.2. Metrics.
  
  * 3.16a.3. CVL and IAM evaluation:
  
    | Non-identical | Criterion | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | Patches | None | 5 | 0.1329 | 0.9579 | 0.9519 | 0.9639 | 0.0361 | 0.0481 | 0.9635 | 0.9525 | 0.9891 |
    |  | IAM | 5 | 0.1164 | 0.9656 | 0.9598 | 0.9714 | 0.0286 | 0.0402 | 0.9711 | 0.9603 | 0.9910 |
    |  | CVL | 5 | 0.1640 | 0.9445 | 0.9419 | 0.9472 | 0.0528 | 0.0581 | 0.9469 | 0.9422 | 0.9844 |
    |  | Hard | 5 | 0.1296 | 0.9560 | 0.9631 | 0.9489 | 0.0511 | 0.0369 | 0.9496 | 0.9626 | 0.9890 |
    | Documents | None | 5 | 0.1445 | 0.9550 | 0.9468 | 0.9632 | 0.0368 | 0.0532 | 0.9626 | 0.9476 | 0.9877 |
    |  | IAM | 5 | 0.1310 | 0.9625 | 0.9548 | 0.9702 | 0.0298 | 0.0452 | 0.9697 | 0.9555 | 0.9893 |
    |  | CVL | 5 | 0.1709 | 0.9427 | 0.9374 | 0.9480 | 0.0520 | 0.0626 | 0.9475 | 0.9380 | 0.9834 |
    |  | Hard | 5 | 0.1445 | 0.9511 | 0.9583 | 0.9440 | 0.0560 | 0.0417 | 0.9448 | 0.9577 | 0.9873 |
    | Languages | None / CVL | 5 | 0.1856 | 0.9382 | 0.9306 | 0.9458 | 0.0542 | 0.0694 | 0.9450 | 0.9317 | 0.9813 |
    |  | Hard | 5 | 0.2186 | 0.9313 | 0.9188 | 0.9439 | 0.0561 | 0.0812 | 0.9424 | 0.9208 | 0.9766 |
    |  | Negative | 5 | 0.0265 | 0.9898 | None | 0.9898 | 0.0102 | None | None | None | None |
       
    * Epochs of model training - EofT - by the best validation accuracy and loss result.
       
* #### 3.16b. Model v2.5.1 evaluation on [256x256] patches with added grid lines (extended train database of raw grayscaled images)

  * 3.16b.1. Database:
    
    * Vide 3.10.1. Database, except for preprocessing v0.5 and addition of grid lines (vide ???);
  
    * Test dataset - 326922 image pairs (equal number of positive and negative instances);
    
    * CVL criterion - 144684 image pairs (equal number of positive and negative instances);
    
    * IAM criterion - 182238 image pairs (equal number of positive and negative instances);
    
    * Hard criterion - 12480 image pairs (equal number of positive and negative instances);
    
    * Negative criterion - 200000 image pairs (an arbitrary number).
       
  * 3.16b.2. Metrics:
  
    * Vide 2.3.2. Metrics.
  
  * 3.16b.3. CVL and IAM evaluation:
  
    | Criterion | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | None | 5 | 1.2496 | 0.7833 | 0.8537 | 0.7129 | 0.2871 | 0.1463 | 0.7483 | 0.8297 | 0.8309 |
    | IAM | 5 | 0.6368 | 0.8261 | 0.7773 | 0.8748 | 0.1252 | 0.2227 | 0.8613 | 0.7971 | 0.9001 |
    | CVL | 5 | 2.7225 | 0.6538 | 0.9509 | 0.3566 | 0.6434 | 0.0491 | 0.5965 | 0.8790 | 0.7125 |
    | Hard | 5 | 2.5697 | 0.6539 | 0.9741 | 0.3337 | 0.6663 | 0.0259 | 0.5938 | 0.9281 | 0.7335 |
    | Negative | 5 | 1.0418 | 0.7777 | None | 0.7777 | 0.2223 | None | None | None | None |
       
    * Epochs of model training - EofT - by the best validation accuracy and loss result.

* #### 3.17. Model 2.6.0 combined evaluation on [256x256] patches (extended train database of raw grayscaled images)

  * 3.17.1. Method of combined evaluation:     

    * For a given pair of documents `XY`, the prediction is based on a combination of all possible partial predictions, i.e. predictions of such `xy` patch pairs, that `x` is a patch of an `X` document and `y` is a patch of an `Y` document;
    
    * Hence all positive and all negative pairs need to be generated (except for pair reverses);
    
    * A given partial prediction is positive if the result is higher than 0.75, and negative if the result is lower than 0.25, else patch prediction is unresolved;
    
    * A given combined prediction is positive if the number of all partial positive predictions is higher than a sum of negative and unresolved partial predictions. Combined negative or unresolved predictions are analogously defined.
  
  * 3.17.2. Model:
  
    * Model v2.5.1 (neither retrained, nor retested via preprocessing v0.6);
    
    * Model training, vide 3.13.3.;
    
    * Model evaluation, vide 3.14.4.;
    
    * However another test dataframe is generated by a method of preprocessing v0.6, for the purpose of combined evaluation. 

  * 3.17.3. Test dataset:
  
      * Preprocessing v0.6;
  
      * A subset of combined CVL and IAM databases, containing 403 document images (189 from CVL and 214 from IAM) by 145 writers (27 from CVL and 118 from IAM); 
      
      * 2970832 image pairs (2886934 negative and 83898 positive instances).
  
  * 3.17.4. Metrics:
  
    * Vide 2.3.2. Metrics, except for Loss and AUC;
    
    * Rate of unresolved instances - `(Number unresolved instances / total number of instances)` - UNR. 
  
  * 3.17.5. CVL and IAM cumulative evaluation of model v2.5.1:
  
    | EofT | Acc | TPR | TNR | FPR | FNR | PPV | NPV | UNR |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | 5 | 0.9899 | 0.9936 | 0.9898 | 0.0102 | 0.0064 | **0.7380** | 0.9998 | 0.0361 |
   
    * Epochs of model training - EofT - by the best validation accuracy and loss result.
    
* #### 3.18. Model v2.6.1
  
  * 3.18.1. Model:
  
    * Exactly the same as v2.5.1, except for much smaller numbers of filters (16, 32 and 64) and minimal stride (1). 
    
  * 3.18.2. Language, libraries and framework / API:
        
    * Vide 2.1.2, but for the TensorFlow v2.1.0.
   
  * 3.18.3. Implementation:
     
    * Google Colaboratory - (2020) - Python 3 Jupyter Notebook, GPU type runtime (Nvidia Tesla K80), 50ms/step (93016 steps per epoch) in the case of model v2.6.1 training on [256x256] patches. 
    
* #### 3.19. Model v2.6.1 training on [256x256] patches (extended train database of raw grayscaled images)

  * 3.19.1. Database:
    
    * Preprocessing v0.6;
      
    * Training dataset - a subset of combined CVL and IAM databases, containing 2740 document images (1415 from CVL and 1325 from IAM) by 822 writers (283 from CVL and 539 from IAM) - 1488246 image pairs (equal number of positive and negative instances) divided into six equal training dataframes;
        
    * Validation dataset - a subset of combined CVL and IAM databases, containing 403 document images (189 from CVL and 214 from IAM) by 145 writers (27 from CVL and 118 from IAM) - 167796 image pairs (equal number of positive and negative instances).
    
     * Because the model is much faster than the v2.5.1, no dataframe needs to be utilized in parts, unless for model tuning.
    
  * 3.19.2. Hyperparameters:
  
    * Vide 3.7.3. Hyperparameters.
  
  * 3.19.3 Trainin:
  
    | Epoch | TDP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/1 | 0.1714 | 0.9314 | 0.3468 | 0.8475 | 0.001 (1e-3) |
    | 2 | 1/6 | 0.1002 | 0.9634 | 0.1404 | 0.9507 | 0.0001 (1e-4) |
    | **3** | 2/6 | 0.0977 | 0.9644 | **0.1320** | **0.9546** | 0.00001 (1e-5) |
    | 4 | 3/6 | 0.0955 | 0.9653 | 0.1372 | 0.9524 | 0.000001 (1e-6) |

    * Training dataframe part - TDP - utilized for a given epoch of training is indicated by its index.

* #### 3.20. Model v2.6.2 training on [256x256] patches (extended train database of raw grayscaled images)

  * 3.20.1 Model:
  
    * Exactly the same as model v2.6.1, but for a dilation of 4 and 2, added to the first and the second convolutional layers, after which no pooling layers are therefore applied.
    
    * Implemented via Google Colaboratory - (2020) - 160ms/step (93016 steps per epoch) in the case of model v2.6.2 training on [256x256] patches.

  * 3.20.1. Database:
    
    * Vide 3.19.1 Database.
    
  * 3.20.2. Hyperparameters:
  
    * Vide 3.7.3. Hyperparameters.
  
  * 3.20.3 Training:
  
    | Epoch | TDP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate | Dropout Rates |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/1 | 0.1855 | 0.9248 | 0.2030 | 0.9213 | 0.001 (1e-3) | 0.5 |
    | 2 | 1/6 | 0.1192 | 0.9550 | 0.1580 | 0.9423 | 0.0001 (1e-4) | 0.5 |
    | **3** | 2/6 | 0.1159 | 0.9564 | **0.1556** | **0.9436** | 0.00001 (1e-5) | 0.5 |
    | 4 | 3/6 | 0.1140 | 0.9577 | 0.1594 | 0.9418 | 0.000001 (1e-6) | 0.5 |
    | 5 | 4/6 | 0.1137 | 0.9571 | 0.1567 | 0.9415 | 0.000001 (1e-5) | 0.1 |
    | 6 | 5/6 | 0.1125 | 0.9580 | 0.1550 | 0.9432 | 0.0000001 (1e-6) | 0.1 |
    | 7 | 6/6 | 0.1127 | 0.9582 | 0.1573 | 0.9435 | 0.00000001 (1e-7) | 0.1 |

    * Training dataframe part - TDP - utilized for a given epoch of training is indicated by its index.
    
* #### 3.21. Model v2.6.3 training on [256x256] patches (extended train database of raw grayscaled images)

  * 3.21.1 Model:
  
    * Exactly the same as model v2.6.1, but for a stride of 4 and 2, added to the first and the second convolutional layers.
    
    * Implemented via Google Colaboratory - (2020) - 40ms/step (93016 steps per epoch) in the case of model v2.6.3 training on [256x256] patches.

  * 3.21.1. Database:
    
    * Vide 3.19.1 Database.
    
  * 3.21.2. Hyperparameters:
  
    * Vide 3.7.3. Hyperparameters.
  
  * 3.21.3 Training:
  
    | Epoch | TDP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/1 | 0.1914 | 0.9215 | 0.2047 | 0.9235 | 0.001 (1e-3) |
    | 2 | 1/6 | 0.1138 | 0.9576 | 0.1748 | 0.9362 | 0.0001 (1e-4) |
    | **3** | 2/6 | 0.1119 | 0.9585 | **0.1638** | **0.9376** | 0.000001 (1e-5) |
    | 4 | 3/6 | 0.1091 | 0.9596 | 0.1733 | 0.9352 | 0.0000001 (1e-6) |

    * Training dataframe part - TDP - utilized for a given epoch of training is indicated by its index.
    
* #### 3.22. Model v2.6.4 training on [256x256] patches (extended train database of raw grayscaled images)

  * 3.22.1 Model:
  
    * Exactly the same as model v2.6.1, but no max-pooling layers are applied.
    
    * Implemented via Google Colaboratory - (2020) - 140ms/step (15503 steps per epoch) in the case of model v2.6.4 training on [256x256] patches.

  * 3.22.1. Database:
    
    * Vide 3.19.1 Database.
    
  * 3.22.2. Hyperparameters:
  
    * Vide 3.7.3. Hyperparameters.
  
  * 3.22.3 Training:
  
    | Epoch | TDP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/6 | 0.3474 | 0.8407 | 0.3976 | 0.8236 | 0.001 (1e-3) |
    | 2 | 2/6 | 0.2505 | 0.8965 | 0.4280 | 0.7904 | 0.001 (1e-3) |
    | 3 | 3/6 | 0.2193 | 0.9119 | 0.5572 | 0.7015 | 0.001 (1e-3) |
    | **4** | 4/6 | 0.1901 | 0.9264 | **0.2092** | **0.9204** | 0.0001 (1e-4) |
    | 5 | 5/6 | 0.1837 | 0.9290 | 0.2068 | 0.9191 | 0.0001 (1e-4) |
    | 6 | 6/6 | 0.1793 | 0.9312 | 0.2107 | 0.9182 | 0.0001 (1e-4) |
    | 7 | 1/6 | 0.1786 | 0.9315 | 0.2117 | 0.9190 | 0.00001 (1e-5) |

    * Training dataframe part - TDP - utilized for a given epoch of training is indicated by its index.
    
* #### 3.23. Model v2.6.5 training on [256x256] patches (extended train database of raw grayscaled images)

  * 3.23.1 Model:
  
    * Model v2.6.5a is exactly the same as model v2.6.1, but for distances passed to the first fully connected layer (hence, the fourth FC layer is the output one), i.e. distances are concatenated with the GAP layers outputs.
    
    * Model v2.6.5b is exactly the same as model v2.6.1, but for distances passed to the fourth fully connected layer (which is therefore the output one), i.e. distances are concatenated with the third FCL outputs, before the last dropout layer.
    
    * Model v2.6.5c is exactly the same as model v2.6.1, but for distances passed to the fourth fully connected layer (which is therefore the output one), i.e. distances are concatenated with the third FCL outputs, after the last dropout layer.
    
    * Implemented via Google Colaboratory - (2020) - 50ms/step (93016 steps per epoch) in the case of model v2.6.5 training on [256x256] patches.

  * 3.23.1. Database:
    
    * Vide 3.19.1 Database.
    
  * 3.23.2. Hyperparameters:
  
    * Vide 3.7.3. Hyperparameters.
  
  * 3.23.3 Model v2.6.5a training:
  
    | Epoch | TDP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/1 | 0.3157 | 0.8644 | 0.3356 | 0.8480 | 0.001 (1e-3) |
    | 2 | 1/1 | 0.2464 | 0.9020 | 0.2442 | 0.9032 | 0.001 (1e-3) |
    | 3 | 1/6 | 0.2207 | 0.9142 | 0.2391 | 0.9042 | 0.0001 (1e-4) |
    
    * Training dataframe part - TDP - utilized for a given epoch of training is indicated by its index.
    
  * 3.23.4 Model v2.6.5b training:
  
    | Epoch | TDP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/1 | 0.3284 | 0.8568 | 0.3004 | 0.8761 | 0.001 (1e-3) |
    | 2 | 1/1 | 0.2458 | 0.9016 | 0.2800 | 0.8886 | 0.001 (1e-3) |
    | 3 | 1/6 | 0.2232 | 0.9127 | 0.2374 | 0.9049 | 0.0001 (1e-4) |
    
    * Training dataframe part - TDP - utilized for a given epoch of training is indicated by its index.
    
  * 3.23.5 Model v2.6.5c training:
  
    | Epoch | TDP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/1 | 0.2127 | 0.9123 | 0.2327 | 0.9105 | 0.001 (1e-3) |
    | 2 | 1/6 | 0.1298 | 0.9529 | 0.1612 | 0.9415 | 0.0001 (1e-4) |
    | **3** | 2/6 | 0.1264 | 0.9545 | **0.1536** | **0.9430** | 0.00001 (1e-5) |
    | 4 | 3/6 | 0.1236 | 0.9560 | 0.1672 | 0.9386 | 0.000001 (1e-6) |

    
    * Training dataframe part - TDP - utilized for a given epoch of training is indicated by its index.
    
* #### 3.24. Model v2.6.6 training on [256x256] patches (extended train database of raw grayscaled images)

  * 3.24.1 Model:
  
    * Exactly the same as model v2.6.1, but convolutional layers are doubled.
    
    * Implemented via Google Colaboratory - (2020) - 360ms/step (93016 steps per epoch) in the case of model v2.6.6 training on [256x256] patches.

  * 3.24.1. Database:
    
    * Vide 3.19.1 Database.
    
  * 3.24.2. Hyperparameters:
  
    * Vide 3.7.3. Hyperparameters.
  
  * 3.24.3 Training:
  
    | Epoch | TDP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/1 | 0.1858 | 0.9247 | 0.2874 | 0.8877 | 0.001 (1e-3) |
    | 2 | 1/6 | 0.1046 | 0.9608 | 0.1554 | 0.9431 | 0.0001 (1e-4) |
    | **3** | 2/6 | 0.1005 | 0.9631 | **0.1475** | **0.9452** | 0.00001 (1e-5) |
    | 4 | 3/6 | 0.0983 | 0.9638 | 0.1551 | 0.9422 | 0.000001 (1e-6) |

    * Training dataframe part - TDP - utilized for a given epoch of training is indicated by its index.

    
* #### 3.25. Model v2.7.0 training on [256x256] patches (extended train database of raw grayscaled images)

  * 3.25.1 Model:
  
    * Exactly the same as model v2.6.1.

  * 3.25.1. Database:
  
    * Preprocessinh v0.7;
    
    * For the purpose of model training, ten times more negative - than positive - instances are generated. 
    
    * Training dataset - a subset of combined CVL and IAM databases, containing 2740 document images (1415 from CVL and 1325 from IAM) by 822 writers (283 from CVL and 539 from IAM) - 8185353 image pairs (744123 positive and 7441230 negative instances) divided into 12 equal training dataframes;
        
    * Validation dataset - a subset of combined CVL and IAM databases, containing 403 document images (189 from CVL and 214 from IAM) by 145 writers (27 from CVL and 118 from IAM) - 167796 image pairs (equal number of positive and negative instances).

  * 3.25.2. Hyperparameters:
  
    * Vide 3.7.3. Hyperparameters.
    
    * Positive samples weight is equal to 1;
    
    * Negative samples weight is equal to 0.1 (i.e. a rounded fraction of positive over netagive instances);
  
  * 3.25.3 Training:
  
    | Epoch | TDP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/12 | 0.0570 | 0.8490 | 0.5447 | 0.7266 | 0.001 (1e-3) |
    | 2 | 2/12 | 0.0395 | 0.8985 | 0.4640 | 0.7812 | 0.001 (1e-3) |
    | 3 | 3/12 | 0.0338 | 0.9153 | 0.2245 | 0.9183 | 0.001 (1e-3) |
    | 4 | 4/12 | 0.0301 | 0.9262 | 0.2074 | 0.9242 | 0.001 (1e-3) |
    | 5 | 5/12 | 0.0271 | 0.9343 | 0.1868 | 0.9341 | 0.001 (1e-3) |
    | 6 | 6/12 | 0.0255 | 0.9387 | 0.2001 | 0.9266 | 0.001 (1e-3) |
    | 7 | 7/12 | 0.0241 | 0.9421 | 0.1858 | 0.9323 | 0.001 (1e-3) |
    | 8 | 8/12 | 0.0228 | 0.9456 | 0.1779 | 0.9380 | 0.001 (1e-3) |
    | 9 | 9/12 | 0.0189 | 0.9546 | 0.1547 | 0.9453 | 0.0001 (1e-4) |
    | 10 | 10/12 | 0.0183 | 0.9559 | 0.1587 | 0.9444 | 0.0001 (1e-4) |
    | 11 | 11/12 | 0.0184 | 0.9562 | 0.1591 | 0.9442 | 0.00001 (1e-5) |
    | 12 | 12/12 | 0.0180 | 0.9569 | 0.1585 | 0.9445 | 0.000001 (1e-6) |
    
    * Training dataframe part - TDP - utilized for a given epoch of training is indicated by its index.

* #### 3.26. Model v2.8.0 training on pairs of [512x512] and [256x256] patches (extended train database of raw grayscaled images)

  * 3.26.1 Model:
  
    * Exactly the same as model v2.6.1, but left convolutional channel input is of size [512x512];
    
    * In other words, the model is trained on pairs of [512x512] (left-channel) and [256x256] (right-channel) patches;
    
    * Implemented via Google Colaboratory - (2020) - 110ms/step (36791 steps per epoch) in the case of model v2.8.0 training on pairs of [512x512] and [256x256] patches.

  * 3.26.1. Database:
  
    * Preprocessinh v0.8;
    
    * Training dataset - a subset of combined CVL and IAM databases, containing 2740 document images (1415 from CVL and 1325 from IAM) by 822 writers (283 from CVL and 539 from IAM) - 588642 image pairs (equal number of positive and negative instances);
        
    * Validation dataset - a subset of combined CVL and IAM databases, containing 403 document images (189 from CVL and 214 from IAM) by 145 writers (27 from CVL and 118 from IAM) - 67048 image pairs (equal number of positive and negative instances).

  * 3.26.2. Hyperparameters:
  
    * Vide 3.7.3. Hyperparameters.
  
  * 3.26.3 Training:
  
    | Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate |
    | --- | --- | --- | --- | --- | --- |
    | 1 | 0.2791 | 0.8832 | 0.2291 | 0.9181 | 0.001 (1e-3) |
    | 2 | 0.1590 | 0.9430 | 0.2194 | 0.9234 | 0.001 (1e-3) |
    | 3 | 0.1143 | 0.9606 | 0.1528 | 0.9429 | 0.0001 (1e-4) |
    | 4 | 0.1071 | 0.9633 | 0.1527 | 0.9447 | 0.0001 (1e-4) |
    | 5 | 0.1018 | 0.9652 | 0.1479 | 0.9455 | 0.00001 (1e-5) |
    | 6 | 0.1008 | 0.9656 | 0.1464 | 0.9460 | 0.000001 (1e-6) |
    
### 4. Verificational Model v3

* #### 4.1. Model v3.6.0
  
  * 4.1.1. Model architecture is the same as model v2 architecture, however:
  
    * Core networks consist only of one convolutional layer each, where strides of 4 and kernels of size 32 are utilized;
    
    * Hence no max pooling layer is applied, instead globally average pooling layers are utilized directly instead.
    
  * 4.1.2. Language, libraries and framework / API:
        
    * Vide 2.1.2.
   
  * 4.1.3. Implementation:
       
    * Google Colaboratory - (2019) - Python 3 Jupyter Notebook, GPU type runtime (Nvidia Tesla K80), 130 ms/step (15503 steps per epoch) in the case of model v3.6.0 training on [256x256] patches.   
    
* #### 4.2. Model v3.6.0 training on [256x256] patches (extended train database of raw grayscaled images)
  
  * 4.2.1. Database:
  
    * Vide 3.14.1. Database, except for preprocessing v0.6;
    
    * Preprocessing v0.6;
      
    * Training dataset - a subset of combined CVL and IAM databases, containing 2740 document images (1415 from CVL and 1325 from IAM) by 822 writers (283 from CVL and 539 from IAM) - 1488246 image pairs (equal number of positive and negative instances) divided into six equal training dataframes;
        
    * Validation dataset - a subset of combined CVL and IAM databases, containing 403 document images (189 from CVL and 214 from IAM) by 145 writers (27 from CVL and 118 from IAM) - 167796 image pairs (equal number of positive and negative instances).
    
  * 4.2.2. Hyperparameters:
  
    * Vide 3.13.3. Hyperparameters.
  
  * 4.2.3. Training:
  
    | Epoch | DfP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/6 | 0.5199 | 0.7432 | 0.4188 | 0.8187 | 0.001 (1e-3) |
    | 2 | 2/6 | 0.3785 | 0.8264 | 0.3478 | 0.8607 | 0.001 (1e-3) |
    | 3 | 3/6 | 0.3400 | 0.8493 | 0.3032 | 0.8844 | 0.001 (1e-3) |
    | 4 | 4/6 | 0.3058 | 0.8691 | 0.2809 | 0.8911 | 0.0001 (1e-4) |
    | 5 | 5/6 | 0.2971 | 0.8736 | 0.2756 | 0.8949 | 0.00001 (1e-5) |
    | 6 | 6/6 | 0.2950 | 0.8751 | 0.2715 | 0.8952 | 0.00001 (1e-5) |
    | 7 | 1/6 | 0.2962 | 0.8752 | 0.2740 | 0.8934 | 0.00001 (1e-5) |
    | **8** | 2/6 | 0.2952 | 0.8752 | **0.2642** | **0.9013** | 0.000001 (1e-6) |
    | 9 | 3/6 | 0.2929 | 0.8757 | 0.2728 | 0.8967 | 0.000001 (1e-6) |
    | 10 | 4/6 | 0.2932 | 0.8759 | 0.2714 | 0.8956 | 0.000001 (1e-6) |
    | 11 | 5/6 | 0.2939 | 0.8758 | 0.2727 | 0.8945 | 0.0000001 (1e-7) |
    | 12 | 6/6 | 0.2929 | 0.8764 | 0.2701 | 0.8969 | 0.00000001 (1e-8) |
    
    * Dataframe part - DfP - utilized for a given epoch of training and validation is indicated by its index.
    
* #### 4.3. Model v3.6.1
  
  * 4.3.1. Model:
  
    * Vide model v3.6.0, however, only 32 kernels of size 16, stride 1 and dilatation 2, are utilized;

    * Also, fully connected layers are half as wide as in the case of model v3.6.0. 
    
  * 4.3.2. Language, libraries and framework / API:
        
    * Vide 2.1.2, but for the TensorFlow v2.1.0.
   
  * 4.3.3. Implementation:
     
    * Google Colaboratory - (2020) - Python 3 Jupyter Notebook, GPU type runtime (Nvidia Tesla K80), 90ms/step (93016 steps per epoch) in the case of model v3.6.1 training on [256x256] patches.   
    
* #### 4.4. Model v3.6.1 training on [256x256] patches (extended train database of raw grayscaled images)
  
  * 4.4.1. Database:
  
    * Vide 4.2.1. Database.
    
    * Because the model is much faster than the v3.6.0, no dataframe needs to be utilized in parts, unless for model tuning.
    
  * 4.4.2. Hyperparameters:
  
    * Vide 3.13.3. Hyperparameters.
  
  * 4.4.3. Training:
  
    | Epoch | DfP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate Reductions |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/1 | 0.4617 | 0.7689 | 0.4242 | 0.8111 | None |
    | 2 | 1/1 | 0.4059 | 0.8061 | 0.4093 | 0.8211 | None |
    
     * Dataframe part - DfP - utilized for a given epoch of training and validation is indicated by its index.
     
* #### 4.5. Model v3.6.2
  
  * Exactly the same as v3.6.1 model, however it is trained by means of databases based transfer learning. 
     
* #### 4.6. Model v3.6.2 transfer training on [256x256] patches (extended train database of raw grayscaled images)
  
  * 4.6.1. Database:
  
    * Vide 4.4.1. Database;
    
    * CVL training dataset - 456460 image pairs;
    
    * CVL validation dataset - 73744 image pairs;
    
    * IAM training dataset - 1031786 image pairs;
    
    * IAM validation dataset - 94052 image pairs;
    
  * 4.6.2. Hyperparameters:
  
    * Vide 3.13.3. Hyperparameters.
  
  * 4.6.3. Training:
  
    | Epoch | Database | DfP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate | 
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | 1 | CVL | 1/1 | 0.4316 | 0.7995 | 0.2873 | 0.8862 | 0.001 (1e-3) |
    | 2 | CVL | 1/1 | 0.2279 | 0.9121 | 0.2312 | 0.9079 | 0.001 (1e-3) |
    | 3 | CVL | 1/1 | 0.2003 | 0.9237 | 0.2253 | 0.9091 | 0.001 (1e-3) |
    | 4 | CVL | 1/1 | 0.1865 | 0.9301 | 0.2010 | 0.9207 | 0.0001 (1e-4) |
    | **5** | **CVL** | 1/1 | 0.1846 | 0.9310 | **0.2003** | **0.9209** | 0.00001 (1e-5) |
    | 6 | IAM | 1/1 | 0.2918 | 0.8903 | 0.2965 | 0.8806 | 0.0001 (1e-4) |
    | 7 | IAM | 1/1 | 0.2272 | 0.9094 | 0.2924 | 0.8847 | 0.0001 (1e-4)) |
    | **8** | **IAM** | 1/1 | 0.2190 | 0.9131 | **0.2811** | **0.8894** | 0.00001 (1e-5) |
    | 9 | Both | 1/1 | 0.2601 | 0.8949 | 0.2949 | 0.8797 | 0.0001 (1e-4) |
    | 10 | Both | 1/6 | 0.2520 | 0.8991 | 0.2970 | 0.8776 | 0.00001 (1e-5) |
    | **11** | **Both** | 2/6| 0.2523 | 0.8989 | **0.2898** | **0.8819** | 0.000001 (1e-6) |
    | 12 | Both | 3/6 | 0.2511 | 0.8992 | 0.2926 | 0.8809 | 0.0000001 (1e-7) |
    
     * Database - utilized for a given epoch of training and validation is indicated by its abbreviation, unless both CVL and IAM databases are utilized combined. 
    
     * Dataframe part - DfP - utilized for a given epoch of training and validation is indicated by its index.
     
  * 4.6.4. Errors:
  
     * During the training epochs 2 through 12, an error in loading the saved (between epochs) optimizer states occured, hence, the optimizer was freshly initialized at the beginning of each of those epochs.
     
 * #### 4.7. Model v3.6.3
  
  * Exactly the same as models v3.6.1 and v3.6.2, however its fully connected layers haven't been narrowed down by half (in other words, those are exactly the same as in the case of the v3.6.0 model).
  
  * Also, the convolutional layers weights of the model v3.6.2 (11th epoch of training), are reused and retrained in the model v3.6.3.
  
* #### 4.8. Model v3.6.3 transfer training on [256x256] patches (extended train database of raw grayscaled images)
  
  * 4.8.1. Database:
  
    * Vide 4.4.1. Database;
    
  * 4.8.2. Hyperparameters:
  
    * Vide 3.13.3. Hyperparameters.
  
  * 4.8.3. Training:
  
    | Epoch | DfP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate | 
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/1 | 0.2964 | 0.8710 | 0.2984 | 0.8790 | 0.001 (1e-3) |
    | 2 | 1/6 | 0.2347 | 0.9054 | 0.3150 | 0.8719 | 0.001 (1e-3) |
    | **3** | 2/6 | 0.2293 | 0.9073 | **0.2595** | **0.8937** | 0.0001 (1e-4) |
    | 4 | 3/6 | 0.2253 | 0.9092 | 0.2627 | 0.8929 | 0.00001 (1e-5) |
    | 5 | 4/6 | 0.2244 | 0.9097 | 0.2644 | 0.8926 | 0.000001 (1e-6) |
  
     * Dataframe part - DfP - utilized for a given epoch of training and validation is indicated by its index.
     
* #### 4.9. Model v3.6.4
  
  * Exactly the same as the model v3.6.1, however the receptive fields (kernels / filters) are of size 32.
  
* #### 4.10. Model v3.6.4 training on [256x256] patches (extended train database of raw grayscaled images)
  
  * 4.10.1. Database:
  
    * Vide 4.4.1. Database;
    
  * 4.10.2. Hyperparameters:
  
    * Vide 3.13.3. Hyperparameters.
  
  * 4.10.3. Training:
  
    | Epoch | DfP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate | 
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/1 | 0.3490 | 0.8462 | 0.3441 | 0.8581 | 0.001 (1e-3) |
    | 2 | 1/1 | 0.2761 | 0.8861 | 0.3125 | 0.8718 | 0.0001 (1e-4) |
    | 3 | 1/2 | 0.2723 | 0.8888 | 0.3132 | 0.8714 | 0.00001 (1e-5) |
 
     * Dataframe part - DfP - utilized for a given epoch of training and validation is indicated by its index.
     
 ### 5. Verificational Model v4

* #### 5.1. Model v4.6.0
  
  * 5.1.1. Model architecture is the same as model v2 architecture, however:
  
    * Nine convolutional layers are applied, and the number of filters is doubled every three layers (i.e. 16, 32, 64);
    
    * Kernel size is 3, stride is 1, dilation rate is 2;
    
    * No max pooling layers are applied.
    
  * 5.1.2. Language, libraries and framework / API:
        
    * Vide 2.1.2.
   
  * 5.1.3. Implementation:
       
    * Google Colaboratory - (2019) - Python 3 Jupyter Notebook, GPU type runtime (Nvidia Tesla K80), 600 ms/step (15503 steps per epoch) in the case of model v4.6.0 training on [256x256] patches.   
    
* #### 5.2. Model v4.6.0 training on [256x256] patches (extended train database of raw grayscaled images)
  
  * 5.2.1. Database:
  
    * 3.19.1. Database.
    
  * 5.2.2. Hyperparameters:
  
    * Vide 3.13.3. Hyperparameters.
  
  * 5.2.3. Training:
  
    | Epoch | DfP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/6 | 0.3834 | 0.8166 | 0.4075 | 0.8200 | 0.001 (1e-3) |
    | 2 | 2/6 | 0.2596 | 0.8913 | 0.4675 | 0.7779 | 0.001 (1e-3) |
    | 3 | 3/6 | 0.2152 | 0.9146 | 0.2744 | 0.8945 | 0.001 (1e-3) |
    | 4 | 4/6 | 0.1966 | 0.9228 | 0.3629 | 0.8545 | 0.001 (1e-3) |
    | 5 | 5/6 | 0.1834 | 0.9287 | 0.2502 | 0.9036 | 0.001 (1e-3) |
    | 6 | 6/6 | 0.1736 | 0.9338 | 0.2447 | 0.9017 | 0.001 (1e-3) |
    | 7 | 1/6 | 0.1513 | 0.9431 | 0.2048 | 0.9182 | 0.0001 (1e-4) |
    | 8 | 2/6 | 0.1460 | 0.9453 | 0.1908 | 0.9241 | 0.00001 (1e-5) |
            
    * Dataframe part - DfP - utilized for a given epoch of training and validation is indicated by its index.
    
* #### 5.3. Model v4.6.1 training on [256x256] patches (extended train database of raw grayscaled images)

  * 5.3.1 Model:
  
     * Analogous to the model v4.6.0, i.e.: 18 convolutional layers are utilized; and the number of filters isn't doubled every three layers, but 16 filters are added every three layers; max pooling layers are utilized every three convolutional layers, instead of dilation.
     
     * Google Colaboratory - (2019) - Python 3 Jupyter Notebook, GPU type runtime (Nvidia Tesla K80), ??? ms/step (15503 steps per epoch) in the case of model v4.6.1 training on [256x256] patches.
  
  * 5.3.2. Database:
  
    * 3.19.1. Database.
    
  * 5.3.3. Hyperparameters:
  
    * Vide 3.13.3. Hyperparameters.
  
  * 5.3.4. Training:
  
    | Epoch | DfP | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1/6 | 0. | 0. | 0. | 0. | 0.001 (1e-3) |
    | 2 | 2/6 | 0. | 0. | 0. | 0. | ? |
    | 3 | 3/6 | 0. | 0. | 0. | 0. | ? |
    | 4 | 4/6 | 0. | 0. | 0. | 0. | ? |
    | 5 | 5/6 | 0. | 0. | 0. | 0. | ? |
    | 6 | 6/6 | 0. | 0. | 0. | 0. | ? |
    | 7 | 1/6 | 0. | 0. | 0. | 0. | ? |
    | 8 | 2/6 | 0. | 0. | 0. | 0. | ? |
            
    * Dataframe part - DfP - utilized for a given epoch of training and validation is indicated by its index.
    
* #### ?.?. Model v?.?.? evaluation on ??? patches (extended train database of ??? images)

  * ?.?.1. Database:
    
    * Vide 3.14.1. Database, except for preprocessing v0.6;
  
    * Test dataset - ??? image pairs (equal number of positive and negative instances);
        
    * IAM criterion - ??? image pairs (equal number of positive and negative instances);
    
    * CVL criterion - ??? image pairs (equal number of positive and negative instances);
    
    * Hard criterion - ??? image pairs (equal number of positive and negative instances);
    
    * Negative criterion - ??? image pairs (an arbitrary number).
       
  * ?.?.2. Metrics:
  
    * Vide 2.3.2. Metrics.
  
  * ?.?.3. CVL and IAM evaluation:
  
    | Criterion | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | None | ? | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
    | IAM | ? | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
    | CVL | ? | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
    | Hard | ? | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
    | Negative | ? | 0. | 0. | None | 0. | 0. | None | None | None | None |
   
    * Epochs of model training - EofT - by the best validation accuracy and loss result.   
    
* #### ?.?. Model ?.?.? combined evaluation on ??? patches (extended train database of ??? images)

  * ?.?.1. Test dataset:
  
      * Preprocessing v0.?;
  
      * A subset of combined CVL and IAM databases, containing 403 document images (189 from CVL and 214 from IAM) by 145 writers (27 from CVL and 118 from IAM); 
      
      * ??? image pairs (??? negative and ??? positive instances).
  
  * ?.?.2. Metrics:
  
    * Vide 3.16.4. Metrics. 
  
  * ?.?.3. CVL and IAM cumulative evaluation:
  
    | EofT | Acc | TPR | TNR | FPR | FNR | PPV | NPV | UNR |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | ? | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
   
    * Epochs of model training - EofT - by the best validation accuracy and loss result.
