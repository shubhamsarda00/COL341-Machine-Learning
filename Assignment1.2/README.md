# Assignment 1.2

In this assignment, we implement **multinomial logistic regression** to predict the number of days (1-8) a patient stays in the hospital. We've again used the **SPARCS Hospital dataset** (https://healthdata.gov/State/Hospital-Inpatient-Discharges-SPARCS-De-Identified/nff8-2va3), as done in Assignment 1.1. Preprocessed data can be found in **data** folder of **src** directory.

In first part of the assignment, we implement the classification model using **batch/mini-batch gradient descent** with original features of the dataset such as Total Costs, Ethnicity, Gender, Risk of Mortality, Severity of Illness Code etc. We implement 3 variants of gradient descent:
a) Fixed Learning Rate
b) Adaptive Learning Rate using n<sub>t</sub>


Details of the experiments done and final model selected can be found in **report.pdf**.

More Details on problem statement in **Assignment_1.pdf**.

### Running the Code

1. **Assignment1.2.ipynb**

The code for all the experiments (including the commented out code) can be found in Assignment1.2.ipynb. In the code for Assignment1.2.ipynb, it has been assumed that the directory containing the notebook also contains the **data** folder, similar to the current repository structure.

2. **linear.py**

In linear.py, we manually create these top 300 features to build the model. There are 3 commands for running this script.

a) **python3 linear.py a trainfile.csv testfile.csv outputfile.txt weightfile.txt**

Here, we run OLS to build the model using original features of the dataset. We write the predictions (1 per line) (for **testfile.csv**) and create a line aligned **outputfile.txt**. We also output our weights (including intercept in the very first line) in the **weightfile.txt**.

b) **python3 linear.py b trainfile.csv testfile.csv regularization.txt outputfile.txt weightfile.txt bestparameter.txt**

The parameters are the same as mode ’a’, with the additional Regularization Parameter List (λ) being an input. The **regularization.txt** file should contain values of λ in a single line separated by ','. Sample file can be found in the **src** folder. We perform Ridge regression with 10-Fold Cross validation for all the λ values and report the output and weights as in previous modes. Additionally, it reports the best regularization parameter in the file **bestparameter.txt**.

c) **python3 linear.py c trainfile.csv testfile.csv outputfile.txt**

We create and use our best features (found using LassoLars) and give predictions predictions by using OLS with the same.

### Libraries Required

1. Numpy
2. Pandas
3. Sklearn
4. Scipy
