# Assignment 1.1

In this assignment we explore different types of regression such as **OLS** (ordinary least squares) regression, **Ridge** regression and **Lasso** regression.
We've used the **SPARCS Hospital Dataset** (https://healthdata.gov/State/Hospital-Inpatient-Discharges-SPARCS-De-Identified/nff8-2va3) to predict Total Costs of a hospital patient given 30 features such as Lenght of Stay, Ethnicity, Gender, Risk of Mortality, Severity of Illness Code etc. Preprocessed data can be found in compressed form in **data** folder of **src** directory. 

In first part of the assignment, we implement OLS and Ridge regression from scratch using original features of the dataset. In the next part of the assignment, we experiment with **feature creation & selection** to improve our results. The general idea is to create large number of features (>300) and then use **LassoLars** (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html) to select top 300 features to create our model. Finally, OLS is used with these selected features to give predictions on the test set. 

Details of the experiments done and final model selected can be found in **report.pdf**.

More Details on problem statement in **Assignment_1.pdf**.

### Running the Code


1. **Assignment1.1.ipynb**

The code for all the experiments (including the commented out code) can be found in Assignment1.1.ipynb. In the code for Assignment1.1.ipynb, it has been assumed that the directory containing the notebook also contains the **data** folder, similar to the current repository structure. 

2. **linear.py**

There are 3 commands for running this script.

a) **python3 linear.py a trainfile.csv testfile.csv outputfile.txt weightfile.txt**

Here, we run OLS to build the model using original features of the dataset. We write the predictions (1 per line) (for **testfile.csv**) and create a line aligned **outputfile.txt**. We also output our weights (including intercept in the very first line) in the **weightfile.txt**.

b) **python3 linear.py b trainfile.csv testfile.csv regularization.txt outputfile.txt weightfile.txt bestparameter.txt**

The parameters are the same as mode ’a’, with the additional Regularization Parameter List (λ) being an input. The **regularization.txt** file should contain values of λ in a single line separated by ','. Sample file can be found in the **src** folder. We perform Ridge regression with 10-Fold Cross validation for all the λ values and report the output and weights as in previous modes. Additionally, it reports the best regularization parameter in the file **bestparameter.txt**.

c) **python3 linear.py c trainfile.csv testfile.csv outputfile.txt**

Here, we manually create the top 300 features (found using LassoLars) to build the model and give predictions predictions by using OLS with the same.

### Libraries Required

1. Numpy
2. Pandas
3. Sklearn
4. Scipy
