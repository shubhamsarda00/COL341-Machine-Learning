# Assignment 3

This assignement was purely competitive and held as an in-house **Kaggle Competition** (https://www.kaggle.com/competitions/col341-a3/). In this problem, we train neural networks to classify yoga poses into different classes. **Yoga pose estimation** has multiple applications such as in creating a mobile application for yoga trainer. There are 19 types of Asanas in the dataset and 29K images for training a machine learning model. Training set contains images from 3 camera angles, whereas the private test set contains images from an unseen 4<sup>th</sup> camera angle. This acts as an impediment in achieving high test accuracy.

We experimented with various CNN architectures, data augmentaion, segmentation followed by classification, key point detection followed by classification etc. Finally, achieved maximum accuracy with a **weighted ensemble** of **XceptionNet**, **EfficientNetB4**, **EfficientNetB5** and **ResNet50V2**. 

Achieved **1<sup>st</sup> rank** in the Kaggle competition with a private test accuracy of **82.2%**. Won the competition by a comfortable margin of **4%**. 
Details of the experiments done and final model implemented can be found in **report.pdf**.

More Details on problem statement in **Assignment_3.pdf**.

### Running the Code

1. **Assignment3.ipynb**

It contains the final code submitted on Kaggle. It trains the final model on training data and writes the prediction in required format (specified on Kaggle webpage) for the test data in **submission.csv**.

2. **train.py**

Command to run the script: **train.py train_data.csv model_path**

This scripts builds and trains the final model on **train_data.csv** (complete path including filename) and saves the model(s) in **model_path** directory.

3. **test.py**

Command to run the script: **test.py model_path private_test.csv submission.csv**

It loads the model(s) saved in **model_path** and writes the prediction in required format (specified on Kaggle webpage) for **private_test.csv** in **submission.csv**.

### Libraries Required

1. numpy
2. pandas
3. opencv
4. tensorflow
5. keras
6. multiprocessing
7. shutil


