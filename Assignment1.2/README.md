# Assignment 1.2

In this assignment, we implement **multinomial logistic regression** to predict the number of days (1-8) a patient stays in the hospital. We've again used the **SPARCS Hospital dataset** (https://healthdata.gov/State/Hospital-Inpatient-Discharges-SPARCS-De-Identified/nff8-2va3), as done in Assignment 1.1. Preprocessed data can be found in **data** folder of **src** directory.

In first part of the assignment, we implement the classification model using **batch/mini-batch gradient descent** with original features of the dataset such as Total Costs, Ethnicity, Gender, Risk of Mortality, Severity of Illness Code etc. We implement 3 variants of gradient descent:
a) Fixed Learning Rate 

b) Adaptive Learning Rate using **n<sub>t</sub> = n<sub>0</sub>/t<sup>0.5</sup>**,  where t= number of iteration

c) Adaptive Learning Rate using **αβ backtracking line search** algorithm

In next part of the assignment, we find the optimal learning rate strategy and the corresponding hyperparamters including batch size to obtain the best results using original features. We further extend this part by incorporating feature creation and selection.

Details of the experiments done and final model selected can be found in **report.pdf**.

More Details on problem statement in **Assignment_1.pdf**.

### Running the Code

1. **Assignment1.2.ipynb**

The code for all the experiments (including the commented out code) can be found in Assignment1.2.ipynb. This includes experiments for finding best learning strategy including hyperparameters as well those pertaining to feature creation and selection. In the code for Assignment1.2.ipynb, it has been assumed that the directory containing the notebook also contains the **data** folder, similar to the current repository structure.

2. **logistic_features_selection.py**

It can be run using the following command: **python3 logistic.py a trainfile.csv testfile.csv param.txt outputfile.txt weightfile.txt**

It first creates the features (>=500) pertaining to the best model. It further selects the most predictive 500 features using **ANOVA** to build the classification model. We write the predictions (1 per line) and create a line aligned outputfile.txt, where the first line will correspond to the first row in testfile.csv and so on. We also output your weight matrix (which includes bias terms in the first row) by flattening the weight matrix rowwise, i.e., write the first row first (1 value per line), then second row and so on and create a line aligned weightfile.txt.

Note: The optimal features to be created were found using experimentation in "Part D" section of Assignment1.1.ipynb

3. **logistic.py**

There are 4 modes (a,b,c,d) for running this script.

a) **python3 logistic.py a trainfile.csv testfile.csv param.txt outputfile.txt weightfile.txt**

Here, param.txt will contain three lines of input, the first being a number (1-3) indicating which learning rate strategy to use and the second being the fixed learning rate (for (i)), η<sub>0</sub> value for adaptive learning rate (for (ii)) or a comma separated (learning rate, α, β) value for αβ backtracking (for (iii)). The third line will be the exact number of epochs for your gradient updates. We use batch gradient descent using original features of the dataset with the specification provided in param.txt and generate output and weights file as done in logistic_features_selection.py.

b) **python3 logistic.py b trainfile.csv testfile.csv param.txt outputfile.txt weightfile.txt**

Here, we perform mini batch gradient descent using the original features of the dataset. The arguments mean the same as mode a, with an additional line 4 in param.txt specifying the batch size (int). 

c) **python3 logistic.py c trainfile.csv testfile.csv outputfile.txt weightfile.txt**

Here, we use the optimal learning rate strategy and corresponding hyperparameters found using experimentation in "Part C" section of Assignment1.1.ipynb file. We again output the outputfile and weightfile as before. For this part, the code would be given 10 minutes to run, and then killed (assignment constraints). No additonal features are created for this mode.

d) **python3 logistic.py d trainfile.csv testfile.csv outputfile.txt weightfile.txt**

Here, we create the features selected using ANOVA in logistic_features_selection.py. We use the optimal learning rate strategy and corresponding hyperparameters found using experimentation in "Part D" section of Assignment1.1.ipynb file. We again output the outputfile and weightfile as before. For this part, the code would be given 15 minutes to run, and then killed (assignment constraints). 

### Libraries Required

1. Numpy
2. Pandas
3. Sklearn
4. Scipy
5. Matplotlib
6. category_encoders (https://contrib.scikit-learn.org/category_encoders/)
