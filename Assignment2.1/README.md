# Assignment 2.1

In this assignment, we implement **Artificial Neural Networks** from scratch to solve classification problems. The assignment can be broken down into 4 parts:

**Part A**: We build a general neural network architecture and train it on a **Toy Dataset** (can be found in **data** folder of **src** directory) to solve the binary classification problem. Loss functions can be one of **mean squared error (MSE)** or **cross entropy loss**. Activation function in hidden layers can be one of **sigmoid**, **tanh** or **relu**. **Xavier initialisation** is used to initialize the weights. Learning rate can be fixed or adaptive.

**Part B**: We modify the neural network implemented in Part A to cater to multiclass classification. The model is trained on the **Devanagri Handwritten Character Dataset** (https://owncloud.iitd.ac.in/nextcloud/index.php/s/TnQqxF4oo6sT2xk). The training dataset has 8-bit 32x32 greyscale images corresponding to 46
Devanagari characters (last column in both training and test data is for labels). 

**Part C**: In Part A and B, we've used convential gradient descent to train the neural networks. In this part, we additionally implement **Momentum**, **Nesterov Accelerated Gradient Descent**, **RMSProp**, **Adam** and **Nadam** optimizers. Further, we are given two generic architectures (\[Input,256,46]  and

a) Fixed Learning Rate 

b) Adaptive Learning Rate using **n<sub>t</sub> = n<sub>0</sub>/t<sup>0.5</sup>**,  where t= number of iteration

c) Adaptive Learning Rate using **αβ backtracking line search** algorithm

In next part of the assignment, we find the optimal learning rate strategy and the corresponding hyperparameters including batch size to obtain the best results using original features. We further extend this part by incorporating feature creation and selection.

Details of the experiments done and final models selected can be found in **report.pdf**.

More Details on problem statement in **Assignment_2.pdf**.

### Running the Code

**Note:** The names of ".csv" files present in the compressed folders should not be modified.

1. **Assignment2.1.ipynb**

The code for all the experiments (including the commented out code) can be found in Assignment2.1.ipynb. This includes experiments for Part C as well as Part D. In the code for Assignment2.1.ipynb, it has been assumed that the directory containing the notebook also contains the **data** folder (similar to the current repository structure), where all the ".csv" files are present. 


2. **neural_a.py**

It can be run using the following command: **python3 logistic.py a trainfile.csv testfile.csv param.txt outputfile.txt weightfile.txt**

It first creates the features (>=500) pertaining to the best model. It further selects the most predictive 500 features using **ANOVA** to build the classification model. We write the predictions (1 per line) and create a line aligned outputfile.txt, where the first line will correspond to the first row in testfile.csv and so on. We also output your weight matrix (which includes bias terms in the first row) by flattening the weight matrix rowwise, i.e., write the first row first (1 value per line), then second row and so on and create a line aligned weightfile.txt.

Note: The optimal features to be created were found using experimentation in "Part D" section of Assignment1.1.ipynb

3. **neural_b.py**



4) **neural_c.py**

Here, param.txt will contain three lines of input, the first being a number (1-3) indicating which learning rate strategy to use and the second being the fixed learning rate (for (i)), η<sub>0</sub> value for adaptive learning rate (for (ii)) or a comma separated (learning rate, α, β) value for αβ backtracking (for (iii)). The third line will be the exact number of epochs for your gradient updates. We use batch gradient descent using original features of the dataset with the specification provided in param.txt and generate output and weights file as done in logistic_features_selection.py.

5) **neural_d.py**

Here, we perform mini batch gradient descent using the original features of the dataset. The arguments mean the same as mode a, with an additional line 4 in param.txt specifying the batch size (int). 



Here, we use the optimal learning rate strategy and corresponding hyperparameters found using experimentation in "Part C" section of Assignment1.1.ipynb file. We again output the outputfile and weightfile as before. For this part, the code would be given 10 minutes to run, and then killed (assignment constraints). No additonal features are created for this mode.


Here, we directly create the features selected using ANOVA in logistic_features_selection.py to build the model. We use the optimal learning rate strategy and corresponding hyperparameters found using experimentation in "Part D" section of Assignment1.1.ipynb file. We again output the outputfile and weightfile as before. For this part, the code would be given 15 minutes to run, and then killed (assignment constraints). 

### Libraries Required

1. Numpy
2. Pandas
4. Scipy
5. Matplotlib



### Helpful Resources 

1. https://ruder.io/optimizing-gradient-descent/index.html
2. https://sudeepraja.github.io/Neural/
3. https://www.ics.uci.edu/~pjsadows/notes.pdf


