# Assignment 2.1

In this assignment, we implement **Artificial Neural Networks** from scratch to solve classification problems. The assignment can be broken down into 4 parts:

**Part A**: We build a general neural network architecture and train it on a **Toy Dataset** (can be found in **data** folder of **src** directory) to solve the binary classification problem. Loss function can be one of **mean squared error (MSE)** or **cross entropy loss**. Activation function in hidden layers can be one of **sigmoid**, **tanh** or **relu**. **Xavier initialisation** is used to initialize the weights. Learning rate can be fixed or adaptive.

**Part B**: We modify the neural network implemented in Part A to cater to multiclass classification. The model is trained on the **Devanagri Handwritten Character Dataset** (https://owncloud.iitd.ac.in/nextcloud/index.php/s/TnQqxF4oo6sT2xk). The training dataset has 8-bit 32x32 greyscale images corresponding to 46
Devanagari characters (last column in both training and test data is for labels). 

**Part C**: In Part A and B, we've used convential gradient descent to train the neural networks. In this part, we additionally implement **Momentum**, **Nesterov Accelerated Gradient Descent**, **RMSProp**, **Adam** and **Nadam** optimizers. Further, we are given two generic architectures: \[Input,256,46]  and \[Input, 512,256,128,64,46]. We tune the hyperparameters such as batch size, learning rate, type of activation function, optimizer algorithm used etc. to minimize the error for both the architectures on the Devanagri dataset. 

**Part D**: In this part, we experiment with different architectures, hyperparameters etc. to maximize our accuracy on the test set of Devanagri dataset.

Details of the experiments done and final models selected can be found in **report.pdf**.

More Details on problem statement in **Assignment_2.pdf**.

### Running the Code

**Note:** The names of ".csv" files present in the compressed folders should not be modified.

1. **Assignment2.1.ipynb**

The code for all the experiments (including the commented out code) can be found in Assignment2.1.ipynb. This includes experiments for Part C as well as Part D. In the code for Assignment2.1.ipynb, it has been assumed that the directory containing the notebook also contains the **data** folder (similar to the current repository structure), where all the ".csv" files are present. 


2. **neural_a.py**

Command to run the script: **python neural_a.py input_path output_path param.txt**

The input training and test files for the toy dataset(filenames will remain same as provided) are present in **input_path** directory (closing '\' also present in the path). We initialise a neural network with the parameters provided in the **param.txt** file (absolute path to param.txt including the filename) and write weights and predictions to **output_path** directory (closing '\' also present in the path).

Note: param.txt will contain 8 lines specifying epochs, batch size, a list specifying the architecture([100,50,10] implies 2 hidden layers with 100 and 50 neurons and 10 neurons in the output layer), learning rate type(0 for fixed and 1 for adaptive), learning rate value, activation function(0 for log sigmoid, 1 for tanh, 2 for relu), loss function(0 for CE and 1 for MSE), seed value for the numpy.random.normal used for weights initialization. The order will be the same as here. 3 weights will be written to the output path for each layer in form of numpy arrays in w_l.npy file where l is the index (starting from 1) of the layers 1 to output layer (example: for architecture[100,50,10], the weight files will be w_1.npy, w_2.npy and w_3.npy). The predictions will be a 1-D numpy array written to the output path as predictions.npy file. Sample parameter files are available in **sample_params** folder of **src** directory.


3. **neural_b.py**

Command to run the script: **python neural_b.py input_path output_path param.txt**

Everything is same as **neural_a** except **input_path** contains the Devanagri dataset files.


4) **neural_c.py**

Command to run the script: **python neural_c.py input_path output_path param.txt**

Here, we use the optimal learning rate strategy and corresponding hyperparameters found using experimentation in "Part C" section of Assignment1.1.ipynb file. We again output the outputfile and weightfile as before. For this part, the code would be given 10 minutes to run, and then killed (assignment constraints). No additonal features are created for this mode.


5) **neural_d.py**

Command to run the script: **python neural_d.py input_path output_path**




Here, we directly create the features selected using ANOVA in logistic_features_selection.py to build the model. We use the optimal learning rate strategy and corresponding hyperparameters found using experimentation in "Part D" section of Assignment1.1.ipynb file. We again output the outputfile and weightfile as before. For this part, the code would be given 15 minutes to run, and then killed (assignment constraints). 

### Libraries Required

1. Numpy
2. Pandas
3. Matplotlib



### Helpful Resources 

1. https://ruder.io/optimizing-gradient-descent/index.html
2. https://sudeepraja.github.io/Neural/
3. https://www.ics.uci.edu/~pjsadows/notes.pdf


