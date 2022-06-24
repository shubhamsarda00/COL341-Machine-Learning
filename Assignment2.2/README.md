# Assignment 2.2

In this assignment, we implement **Convoluional Neural Networks** using **PyTorch** library. The assignment can be broken down into 3 parts:

**Part A**: We build the CNN architecture mentioned in the problem statement and train it on the **Devanagri Character Recognition Dataset** (https://owncloud.iitd.ac.in/nextcloud/index.php/s/TnQqxF4oo6sT2xk). The training dataset has 8-bit 32x32 greyscale images corresponding to 46
Devanagari characters (last column in both training and test data is for labels). Training parameters specified in the problem statement were followed. The results were compared with those obtained by the best ANN architecture found in part D of Assignment 2.1.

**Part B**: We build another CNN architecture mentioned in the problem statement and train it on the **CIFAR-10 Dataset** (https://owncloud.iitd.ac.in/nextcloud/index.php/s/jwFwJSETBGp4MGn). The training dataset has 8-bit 32x32 RGB images corresponding to 10 classes (First column (index=0) contains labels and rest 3072 columns of image data). Training parameters specified in the problem statement were followed. The results were compared with those obtained by the best ANN architecture found in part D of Assignment 2.1.

**Part C**: This is the competitive part of the assignment. Here, we experiment with various CNN architectures, feature engineering techniques, optimizers, learning rates etc. to build a model with less than 3 million parameters (including both trainable and non-trainable) which achieves maximum accuracy on the private test set of CIFAR-10 Dataset. The private test set was only released after the assignment deadline. 

Created a novel model (<3M Params) with a variant of the fused MBConv block, mentioned in https://arxiv.org/pdf/2104.00298v3.pdf, as basic building block of the CNN architecture. Achieved **2<sup>nd</sup> rank** in the whole batch with an accuracy of **89.5%** on the private test set (4000 samples).

![image](https://user-images.githubusercontent.com/45795080/175501636-1cbb7588-e2bc-4d8e-bd31-af6ff4289d78.png)
 

Details of the experiments done and final model implemented can be found in **report.pdf**.

More Details on problem statement in **Assignment_2.pdf**.

### Running the Code

**Note:** The names of ".csv" files present in the compressed folders should not be modified.

1. **Assignment2.2.ipynb**

The code for all the experiments (including the commented out code) can be found in Assignment2.2.ipynb. This includes experiments for Part C. In the code for Assignment2.2.ipynb, it has been assumed that the directory containing the notebook also contains the **cifar10_data** and **devanagri_data** folders, where the corresponding ".csv" files are present. 


2. **train_a.py**

Command to run the script: **python train_a.py train_data.csv test_data.csv model.pth loss.txt  accuracy.txt**

It trains the model associated with part A on the Devanagri dataset (complete path including filename in **train_data.csv**) and dumps the **model.pth** file in the same directory as your .py script. It further keeps track of average training loss for each epoch and test accuracy after each epoch wrt to the public test set  (complete path including filename in **test_data.csv**) containing labels in the last column. It then writes these loss and accuracy values for every epoch in a separate line in **loss.txt** and **accuracy.txt** respectively. 

3. **train_b.py**

Command to run the script: **python train_b.py train_data.csv test_data.csv model.pth loss.txt  accuracy.txt**

It trains the model associated with part B on the CIFAR-10 dataset (complete path including filename in **train_data.csv**) and dumps the **model.pth** file in the same directory as your .py script. It further keeps track of average training loss for each epoch and test accuracy after each epoch wrt to the public test set  (complete path including filename in **test_data.csv**) containing labels in the first column. It then writes these loss and accuracy values for every epoch in a separate line in **loss.txt** and **accuracy.txt** respectively. 

4) **train_c.py**

Command to run the script: **python train_c.py train_data.csv test_data.csv model.pth loss.txt  accuracy.txt**

It trains the model associated with part C on the CIFAR-10 dataset (complete path including filename in **train_data.csv**) and dumps the **model.pth** file in the same directory as your .py script. It further keeps track of average training loss for each epoch and test accuracy after each epoch wrt to the public test set  (complete path including filename in **test_data.csv**) containing labels in the first column. It then writes these loss and accuracy values for every epoch in a separate line in **loss.txt** and **accuracy.txt** respectively. 

We use the best model architecture and optimal training specifications found using experimentation in "Part C" section of Assignment2.2.ipynb file. Details of the experiments done and final model implemented can be found in **report.pdf**.


5) **test_a.py**

Command to run the script: **python test_a.py private_test_data.csv trained_model.pth pred.txt**

It loads the pretrained model weights (using torch.load()) corresponding to the model in part A and generates the predictions corresponding to the given test data (complete path including filename in **private_test_data.csv**). The prediction labels for test data samples are written to **pred.txt**, each in separate line. 

6) **test_b.py**

Command to run the script: **python test_b.py private_test_data.csv trained_model.pth pred.txt**

It loads the pretrained model weights (using torch.load()) corresponding to the model in part B and generates the predictions corresponding to the given test data (complete path including filename in **private_test_data.csv**). The prediction labels for test data samples are written to **pred.txt**, each in separate line. 

7) **test_c.py**

Command to run the script: **python test_a.py private_test_data.csv trained_model.pth pred.txt**

It loads the pretrained model weights (using torch.load()) corresponding to the model in part C and generates the predictions corresponding to the given test data (complete path including filename in **private_test_data.csv**). The prediction labels for test data samples are written to **pred.txt**, each in separate line.  

**Note**: **private_test_data.csv** for all the test scripts should **not** contain the label column. Pretrained weights for models can be found in **pretrained_weights** folder of **src** directory. 

### Libraries Required

1. Numpy
2. Pandas
3. Pytorch (>=1.9.1.post3)

### Helpful Resources

1. https://www.analyticsvidhya.com/blog/2019/09/introduction-to-pytorch-from-scratch/
2. https://www.youtube.com/watch?v=c36lUUr864M
