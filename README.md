## Verificational Model

### 0. Introduction

* #### 0.1. The objective of author's repository was to introduce a method for verification of offline handwritten documents authorship by artificial neural networks (through classification of preprocessed image pairs to positive `same author` or negative `different authors` class).

* #### 0.2. Author's assumptions were that:
     
    * Document images will be best processed simultaneously by two separate convolutional neural networks and classified by one multilayer perceptron (fully connected layers);       
    
    * Preprocessing shouldn't drastically affect image quality (*e.g* by image binarization) to preserve most of handwriting features.

* #### 0.3. Versioning

   * 0.3.1 Experiment identifiers (*e.g* vX.Y.Z):

      * X indicates a model version;
      * Y indicates a method of preprocessing;
      * Z indicates any extra variation of a given X.Y base combination.

   * 0.3.2 Model version identifiers (*e.g* vX):

      * X indicates a model version (as already stated in 0.3.1).

   * 0.3.3 Preprocessing method identifiers (*e.g* v0.Y):
    
      * X is zeroed and Z skipped to avoid any confusion of model versions and preprocessing methods; 
      * Y indicates a methon of preprocessing (as already stated in 0.3.1).

* #### 0.3. Database (9455 documents by 2911 writers):
       
    * Dataset of 1604 documents (full page scans) from CVL database (310 writers), by: F. Kleber, S. Fiel, M. Diem, R. Sablatnig, *CVL-Database: An Off-line Database for Writer Retrieval, Writer Identification and Word Spotting*, "In Proc. of the 12th Int. Conference on Document Analysis and Recognition (ICDAR) 2013" 2013, p. 560 - 564;
        
    * Dataset of 4704 documents (full page scans) from CEDAR (Center of Excellence for Document Analysis and Recognition) database (1568 writers), by: S. Srihari, S. Cha, H. Arora, S. Lee, *Individuality of Handwriting*, "Journal of Forensic Sciences" 2002, No. 4 (Vol. 47), p. 1 - 17;
        
    * Dataset of 1539 documents (full page scans) from IAM (offline handwritten documents) database (657 writers), by: U. Marti, H. Bunke, *The IAM-database: An English Sentence Database for Off-line Handwriting Recognition*, "Int'l Journal on Document Analysis and Recognition" 2002, No. 5, p. 39 - 46;
        
    * Dataset of 208 documents (full page scans) from ICDAR 2011 Writer Identification Contest database (26 writers), by: G. Louloudis, N. Stamatopoulos, B. Gatos, *ICDAR 2011 Writer Identification Contest*, "2011 International Conference on Document Analysis and Recognition" 2011, p. 1475 - 1479;
        
    * Dataset of 400 documents (cropped page scans) from ICFHR 2012 Competition on Writer Identification (Challenge 1: Latin/Greek Documents) database (100 writers), by: G. Louloudis, N. Stamatopoulos, B. Gatos, *ICFHR2012 Competition on Writer Identification Challenge 1: Latin/Greek Documents*, "2012 International Conference on Frontiers in Handwriting Recognition" 2012, p. 825 - 830;
        
    * Dataset of 1000 documents (cropped page scans) from ICDAR 2013 Competition on Writer Identification database (250 writers), by: G. Louloudis, N. Stamatopoulos, B. Gatos, *ICDAR 2013 Competition on Writer Identification*, "2013 12th International Conference on Document Analysis and Recognition" 2013, p. 1397 - 1041.

### 1.1. Preprocessing v0.1 (CVL database, grayscaled images)
              
* #### 1.1.1. Steps of preprocessing in the case of CVL database:
               
    * Step one `CVL_Images_v0.1.py` - conversion of images (scans of whole documents) to grayscale (scale from black = 0 to white = 255), color inversion, extraction of writing space from images, reduction of extracts dimensions to [1024x1024] pixels, division of extracts into [256x256] pixel patches, conversion from the `tif` to `png` format. Patches which do not contain or contain a small amount of text are skipped by the program because of the arbitrary average pixel value threshold - in any case, patches can be sorted by their size and manually removed on that basis;
            
    * Step two `Dataframe_v0.1.py` - creation of a dataframe (a `csv` file that can be edited in any spreadsheet program, *e.g.* calc / excel) separately for the test and training subset, by combinatorial paring of image names into the positive class, and a random combinatorial paring of image names into the negative class (the number of possible negative combinations is much greater than the number positive ones, so all positive instances are created first and then the negative instances are randomly combinated until their number is equal to the number of positive instances). Image name pairs and their labels are ordered by rows, according to columns `left convolutional path`, `right convolutional path`, and `label` (labels are determined by the convergence or divergence of author's identifiers - *e.g.* first four digits of a raw image name in the case of a CVL database). Above method requires that the test and training images are kept in different directories during their preprocessing. However, it is not necessary to create manually any of dataframe `csv` files, *i.e.* they will be created by the program (and if any such a file was already created manually, its directory and name has to be indicated in the program code). Validation dataframe (utilized only for testing of the model during its training, generally after every epoch) is also created, by random sampling of the test dataframe instances (fratcion of which to pull has to be specified - usually 0.1 / 0.2 is enough for validation purposes). Due to the randomness of sampling, it is most probable that the number of sampled positive and negative instances will be effectively equal.
               
* #### 1.1.2. Preprocessing programs
   		
  * 1.1.2.1. Programming language - Python 3.7.3.
  
  * 1.1.2.2. Libraries - OpenCV 4.1.0, Numpy 1.16.4, tqdm 4.33.0, and other common python libraries.
  
  * 1.1.2.3. To utilize preprocessing programs:
  
    * Install Python and listed libraries (installation method depends on user's operating system);
    * Download the repository;
    * Using any given text editor, access programs code and edit image/dataframe input/output paths (save the program files in a `py` format); 
    * Access the directory - which contains a given preprocessing program file - through the terminal / command-line interpreter (method  of access depends on user's operating system);
    * In the terminal type the command `python3 program_name.py` to run the named program;
    * If it were necessary: to force quit (terminate) a running program use a keyboard shortcut `Ctrl + C` in an opened terminal window, or `Ctrl + Z` to suspend a running program, then to resume a paused run, type the command `fg` (works in terminals of most operating systems, *e.g.* macOS, Linux).

### 1.2. Preprocessing v0.2 (CVL database, binarized images)

* #### 1.2.1. In the case of CVL database, method of preprocessing is exactly the same as v0.1, except for:

    * Image binarization (Otsu's method), *ergo* higher threshold of mean pixel value is applied;
    
    * Extraction window is slightly shifted to better fit the writting space, hence greater number of image patches is passed through the mean pixel value threshold.

* #### 1.2.2. Dataframes are generated exatly the same way as in the case of v0.1.

### 1.3. Preprocessing v0.3 (CVL and IAM database, binarized images)

* #### 1.3.1. In the case of CVL database, method of preprocessing is exactly the same as v0.2.

* #### 1.3.2. In the case of IAM database, method of preprocessing is exactly the same as v0.2, however following additional steps were applied:
      
    *
    *
    *

* #### 1.3.3. Dataframes are generated exatly the same way as in the case of v0.2 and v0.1, except for the optional split of training dataframes into a given number of smaller equal size training dataframes, due to the sheer number of training instances.
  
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
      
      * 82476 image pairs (41238 positive and 41238 negative instances);
      
      * Custom hard criterion - excluded documents containing the same samlpe text as training documents (*ergo* included documents containing only samlpe texts no. 7 and 8) - 7572 image pairs (3786 positive and 3786 negative instances).
      
    * IAM database:
      
      * Whole database as a test set of 1539 document images by 657 writers;
      
      * 589274 image pairs only, due to a high background noise and thefore a threshold of mean pixel value >= 16 (294637 positive and 294637 negative instances);
      
      * 590000 roughly denoised (by thresholding of pixel values lower than 55 to 0) images pairs (295000 positive and 295000 negative instances).
        
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
      
  * 2.3.3. CVL evaluation (epochs of model training - EofT - by the best validation result):
     
  | Criterion | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | False | 5 | 0.2331 | 0.9124 | 0.9207 | 0.9040 | 0.0959 | 0.0792 | 0.9056 | 0.9194 | 0.9676 |
  | **True** | 5 | **0.1795** | **0.9370** | 0.9617 | 0.9123 | 0.0876 | 0.0382 | 0.9164 | 0.9597 | 0.9781 |
              
  * 2.3.4. IAM evaluation (epochs of model training - EofT - by the best validation result):
       
  | Denoised | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | **False** | 5 | 265.3726 | 0.5047 | **0.0113** | **0.9979** | 0.0020 | 0.9886 | 0.8478 | 0.5023 | 0.5058 |
  | True | 5 | 0.3629 | 0.8835 | 0.9661 | 0.8009 | 0.1990 | 0.0338 | 0.8291 | 0.9594 | 0.9603 |

* #### 2.4. Model v1.1.1 training on [512x512] patches
  
  * 2.4.1. Database:
  
    * Preprocessing v0.1 ([1024x1024] extracts split into 4 patches instead of 16);
       
    * Training dataset - CVL database subset of 1415 document images by 283 writers - 59868 image pairs (29934 positive and 29934 negative instances);
        
    * Validation dataset - CVL database subset of 189 document images by 27 writers - 2436 image pairs (1201 positive and 1235 negative instances).

  * 2.4.2. Callbacks: 
  
    * *Vide* 2.2.2. Callbacks.
  
  * 2.4.3. Hyperparameters: 
    
    * *Vide* 2.2.3. Hyperparameters.
   
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
       
    * Training dataset - CVL database subset of 1415 document images by 283 writers - 14150 image pairs (7075 positive and 7075 negative instances);
        
    * Validation dataset - CVL database subset of 189 document images by 27 writers - 646 image pairs (319 positive and 327 negative instances).

  * 2.5.2. Callbacks: 
  
    * *Vide* 2.2.2. Callbacks.
  
  * 2.5.3. Hyperparameters: 
    
    * *Vide* 2.2.3. Hyperparameters;
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
  
  * 3.1.1. Model architecture:
  
    * .
    
  * 3.1.2. Language, libraries and framework / API:
        
    * *Vide* 2.1.2.
   
  * 3.1.3. Implementation:
       
    * Google Colaboratory - (2019) - Python 3 Jupyter Notebook, GPU type runtime (Nvidia Tesla K80), 250ms/step (27732 steps per epoch) in case of model v2.1.0 training on [256x256] patches.    
  
* #### 3.2. Model v2.1.0 training on [256x256] patches
 
  * 3.2.1. Database:
  
    * *Vide* 2.2.1. Database.
 
  * 3.2.2. Callbacks: 
  
    * *Vide* 2.2.2. Callbacks.
  
  * 3.2.3. Hyperparameters: 
    
    * *Vide* 2.2.3. Hyperparameters;
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
  
    * *Vide* 2.3.1. Database.
  
  * 3.3.2. Metrics:
  
    * *Vide* 2.3.2. Metrics.
  
  * 3.3.3. CVL evaluation (epochs of model training - EofT - by the best accuracy result):
  
   | Criterion | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | False | 3 | 0.1835 | 0.9387 | 0.9301 | 0.9474 | 0.0526 | 0.0699 | 0.9464 | 0.9312 | 0.9817 |
   | **True** | 3 | **0.1494** | **0.9506** | 0.9567 | 0.9445 | 0.0555 | 0.0433 | 0.9452 | 0.9561 | 0.9858 |
  
  * 3.3.4. IAM evaluation (epochs of model training - EofT - by the best accuracy result):
  
   | Denoised | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | **False** | 3 | 283.7893 | 0.5266 | **0.9883** | **0.0649** | 0.9350 | 0.0116 | 0.5138 | 0.8476 | 0.5269 |
   | True | 3 | 0.6610 | 0.9138 | 0.9867 | 0.8408 | 0.1592 | 0.0133 | 0.8611 | 0.9845 | 0.9510 |
   
* #### 3.4. Model v2.1.1 training on [256x256] patches (simplified variant of model v2.1.0)

  * 3.4.1. Simplified Model:
    
    * Exactly the same as model v2.1.0, except for additional output neuron and distance layers.

  * 3.4.2. Database:
  
    * *Vide* 2.2.1. Database.
 
  * 3.4.3. Callbacks: 
  
    * *Vide* 2.2.2. Callbacks.
  
  * 3.4.4. Hyperparameters: 
    
    * *Vide* 2.2.3. Hyperparameters;
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
    
    * *Vide* 2.2.3. Hyperparameters;
    * Epochs - 5 (five separate runs);
    * Learning rate - initial 0.001 (1e-3), then manually adjusted by a factor of 0.1 after every epoch.
 
  * 3.5.3. Training:
     
    | Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate Reductions |
    | --- | --- | --- | --- | --- |  --- |
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
  
    * *Vide* 2.3.2. Metrics.
  
  * 3.6.3. CVL evaluation (epochs of model training - EofT - by the best accuracy and loss result):
  
   | EofT | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | 4 | 0.2783 | 0.9275 | 0.8941 | 0.9608 | 0.0392 | 0.1059 | 0.9580 | 0.9007 | 0.9749 |
    
* #### 3.?. Model v2.3.0 training on [256x256] patches (extended train database)

  * 3.?.1. Model v2.3.0:
  
    * Exactly the same as v2.1.0, except for training on combained IAM and CVL databases.
     
  * 3.?.2. Database:
  
    * Preprocessing v0.3
      
    * Training dataset - a subset of combined CVL and IAM databases, containing 2740 document images (1415 from CVL and 1325 from IAM) by 822 writers (283 from CVL and 539 from IAM) - 3535056 image pairs (equal number of positive and negative instances);
        
    * Validation dataset - a subset of combined CVL and IAM databases, containing 403 document images (189 from CVL and 214 from IAM) by 145 writers (27 from CVL and 118 from IAM) - 20% of test instances.
      
* #### 3.?. Model v2.3.0 evaluation on [256x256] patches (extended test database)

  * 3.?.3. Database:
  
    * Test dataset - a subset of combined CVL and IAM databases, containing 403 document images (189 from CVL and 214 from IAM) by 145 writers (27 from CVL and 118 from IAM) - 405306 image pairs (equal number of positive and negative instances).  

### 4. Verificational Model v3
