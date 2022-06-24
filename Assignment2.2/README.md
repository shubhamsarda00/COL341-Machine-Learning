# Assignment 2.2

In this assignment, we implement **Convoluional Neural Networks** using **PyTorch** library. The assignment can be broken down into 3 parts:

**Part A**: We build the CNN architecture mentioned in the problem statement and train it on the **Devanagri Character Recognition Dataset** (https://owncloud.iitd.ac.in/nextcloud/index.php/s/TnQqxF4oo6sT2xk). The training dataset has 8-bit 32x32 greyscale images corresponding to 46
Devanagari characters (last column in both training and test data is for labels). Training parameters specified in the problem statement were followed. The results were compared with those obtained by the best ANN architecture found in part D of Assignment 2.1.

**Part B**: We build another CNN architecture mentioned in the problem statement and train it on the **CIFAR-10 Dataset** (https://owncloud.iitd.ac.in/nextcloud/index.php/s/jwFwJSETBGp4MGn). The training dataset has 8-bit 32x32 RGB images corresponding to 10 classes (First column (index=0) contains labels and rest 3072 columns of image data). Training parameters specified in the problem statement were followed. The results were compared with those obtained by the best ANN architecture found in part D of Assignment 2.1.


**Part C**: This is the competitive part of the assignment. Here, we experiment with various CNN architectures, feature engineering techniques, optimizers, learning rates etc. to build a model with less than 3 million parameters (including both trainable and non-trainable) which achieves maximum accuracy on the private test set of CIFAR-10 Dataset. The private test set was only released after the assignment deadline. 

Achieved **2<sup>nd</sup> rank** in the whole batch with an accuracy of **89.5%** on the private test set (4000 samples).

Details of the experiments done and final model implemented can be found in **report.pdf**.

More Details on problem statement in **Assignment_2.pdf**.

### Running the Code

**Note:** The names of ".csv" files present in the compressed folders should not be modified.

1. **Assignment2.2.ipynb**

The code for all the experiments (including the commented out code) can be found in Assignment2.2.ipynb. This includes experiments for Part C. In the code for Assignment2.2.ipynb, it has been assumed that the directory containing the notebook also contains the **cifar10_data** and **devanagri_data** folders, where the corresponding ".csv" files are present. 


2. **train_a.py**

Command to run the script: **python neural_a.py input_path output_path param.txt**

The input training and test files for the toy dataset (filenames will remain same as provided) are present in **input_path** directory (closing \ also present in the path). We initialise a neural network with the parameters provided in the **param.txt** file (absolute path to param.txt including the filename) and write weights and predictions to **output_path** directory (closing \ also present in the path).

Note: param.txt will contain 8 lines specifying epochs, batch size, a list specifying the architecture ([100,50,10] implies 2 hidden layers with 100 and 50 neurons and 10 neurons in the output layer), learning rate type (0 for fixed and 1 for adaptive), learning rate value, activation function (0 for log sigmoid, 1 for tanh, 2 for relu), loss function (0 for CE and 1 for MSE), seed value for the numpy.random.normal used for weights initialization. The order will be the same as here. 3 weights will be written to the output path for each layer in form of numpy arrays in w_l.npy file where l is the index (starting from 1) of the layers 1 to output layer (example: for architecture[100,50,10], the weight files will be w_1.npy, w_2.npy and w_3.npy). The predictions will be a 1-D numpy array written to the output path as predictions.npy file. Sample parameter files are available in **sample_params** folder of **src** directory.


3. **train_b.py**

Command to run the script: **python neural_b.py input_path output_path param.txt**

Everything is same as **neural_a** except **input_path** contains the Devanagri dataset files.


4) **train_c.py**

Command to run the script: **python neural_c.py input_path output_path param.txt**

Here, **input_path** and **ouput_path** remain same as in part B. Here the file **param.txt**, to which the complete absolute path will be provided, will contain 1 line which will be a list specifying the architecture (just the way it is specified in param.txt for part a). The code must not exceed a time limit of 300 seconds. The script trains the specified model with the optimal parameters found and writes the weight files after training to the output path in the same format as in  part a. It also writes a text file to the output path with the name "my params.txt" specifying each of the following in a new line for your best parameters: number of epochs, batch size, learn rate type (0 for fixed 1 for adaptive), learn rate value (initial value in case of adaptive), activation function (0 for log sigmoid, 1 for tanh, 2 for relu), loss function (0 for CE an 1 for MSE), optimizer type (0 for vanilla SGD, 1 for momentum, 2 for nesterov, 3 for RMSprop and 4 for adam, 5 for nadam), seed value for the numpy.random.normal

We use the optimal hyperparameters found for both the architectures using experimentation in "Part C" section of Assignment2.1.ipynb file. 


5) **test_a.py**

Command to run the script: **python neural_d.py input_path output_path**

Here, **input_path** and **ouput_path** remain same as in part B. The script trains the best architecture along with best parameters (found using experimentation in "Part D" section of Assignment2.1.ipynb file). It produces the weight files to output path in the same way as specified for part a). It also write a text file to the output path with the name my params.txt specifying each of the following in a new line for your best parameters:number of epochs, batch size, learn rate type (0 for fixed 1 for adaptive), learn rate value (initial value in case of adaptive), activation function (0 for log sigmoid, 1 for tanh, 2 for relu), loss function (0 for CE an 1 for MSE), optimizer type (0 for vanilla SGD, 1 for momentum, 2 for nesterov, 3 for RMSprop and 4 for adam, 5 for nadam), architecture (list specifying architecture, [100,50,10] implies 2 hidden layers with 100 and 50 neurons and 10 neurons in the output layer), seed value for the numpy.random.normal


6) **test_b.py**

Command to run the script: **python neural_d.py input_path output_path**

7) **test_c.py**

Command to run the script: **python neural_d.py input_path output_path**


### Libraries Required

1. Numpy
2. Pandas
3. Pytorch (>=1.9.1.post3)

### Helpful Resources

1. https://www.analyticsvidhya.com/blog/2019/09/introduction-to-pytorch-from-scratch/
2. https://www.youtube.com/watch?v=c36lUUr864M
