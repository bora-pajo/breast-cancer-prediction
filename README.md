## Predicting malignant versus benign breast cancer cases
Predicting the probability that a diagnosed breast cancer case is malignant or benign based on Wisconsin dataset from UCI repository. 

### Project Overview
According to the Centers for Disease Control and Prevention (CDC) breast cancer is the most common type of cancer for women regardless of race and ethnicity (CDC, 2016). Around 220,000 women are diagnosed with breast cancer each year in the United States (CDC, 2016). Although we may not be aware of all the factors contributing in developing breast cancer, certain attributes such as family history, age, obesity, alcohol and tobacco use have been identified from research studies on this topic (DeSantis, Ma, Bryan, & Jemal, 2014). Breast images procedures such as mammography have been found to be quite effective in early identification cases of breast cancer (Ball, 2012).  When breast images procedures are not utilized, patients can find out late about their diagnosis to be able to treat it.  Similar work on attempting to find the best way to predict the type of cancer based on images of mammograms has identified Support Vector Machine as the best predictor after tuning parameters.  
 

#### Files in this repository
Many files in the respository are examples and things to be saved as the work progresses. Those are names examples_01 and so on. 
The file where you can find all the coding for different classifiers is called _breast_cancer_clean.ipynb_ 

### Libraries used
```python
import numpy as np #for linear algebra
import pandas as pd #for chopping, processing
import csv #for opening csv files
%matplotlib inline 
import matplotlib.pyplot as plt #for plotting the graphs
from scipy import stats #for statistical info
from time import time

from sklearn import tree
from sklearn.model_selection import train_test_split # to split the data in train and test
from sklearn.model_selection import KFold # for cross validation
from sklearn.grid_search import GridSearchCV  # for tuning parameters
from sklearn import metrics  # for checking the accuracy 

#Classifiers 

from sklearn import svm #for Support Vector Machines
from sklearn.svm import SVC # for support vector classifier
from sklearn.neighbors import NearestNeighbors #for nearest neighbor classifier
from sklearn.neighbors import KNeighborsClassifier # for K neighbor classifier
from sklearn.tree import DecisionTreeClassifier #for decision tree classifier
from sklearn.naive_bayes import GaussianNB  #for naive bayes classifier
from sklearn.ensemble import RandomForestClassifier #for Random Forest
from sklearn.ensemble import AdaBoostClassifier # for Ada Boost
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis # for Quadratic Discriminant Analysis
from sklearn.neural_network import MLPClassifier # for multi layer perceptron classifier
```


### Uses
This project tests various classifiers (logistic regression, SVM, decision trees, random forest, and others) to find the classifier that predicts the testing data with the best possible accuracy at the shortest time possible.  

First, it examines the data looking for patterns of correlations among variables. Then it runs various classifiers and compare them. Once a number of 2-3 classifiers is identified as most accurate, the model is fine tuned again. Then, we can see which model predicts data with the highest accuracy.  


### Graphs
Some examples from bokeh and lightning as well as matplolib plots.

### Classifiers used to predict the breast cancer for a training size = 300 

Type of Classifiers | Training Time | Prediction Time| F1 score (training set) | **F1 Score (testing set)**
:---:|:---:|:---:|:---:|:---:
**K-nearest neighbor** | .0012 | .0022 | .9264 | .9081
**Decision Trees** | .0033 | .0002 | 1.000 | .9239
**SVC** | .0051 |.0029 | 1.000 |.0000
**Naive Bayes** |.0006 | .0003 |.9058 | .9341
**Random Forest** | .0381 | .0064 | 1.000 | .9101
**AdaBoost** | .1866 | .0055 | 1.000 | .9451
**QDA** |.0013 | .0005 | .9528 | .9688
**MLP** |.0084 | .0005 | .6648 | .6267


_The F-1 score measures the accuracy of the prediction. The closer it is to a value of 1, the better the prediction is._
QDA seems to work perfectly in this case compared to other forms of classifiers and it is also quite fast to train the data.

A second round of classifiers with an increase training size of 400 was attempted again.  Results are shown below:

### Classifiers used to predict the breast cancer for a training size = 400 

Type of Classifiers | Training Time | Prediction Time| F1 score (training set) | **F1 Score (testing set)**
:---:|:---:|:---:|:---:|:---:
**K-nearest neighbor** | .0006 | .0029 | .9290 | .9038
**Decision Trees** | .0049 | .0003 | 1.000 | .9074
**SVC** | .0110 |.0068 | 1.000 |.0000
**Naive Bayes** |.0007 | .0003 |.9164 |.9541
**Random Forest** | .0403 | .0058 | .9936 | .9358
**AdaBoost** | .1767 | .0057 | 1.000 | .9541
**QDA** |.0013 | .0005 | .9585 | .9381
**MLP** |.0102 | .0005 | .5663| .4843

ROC information added for the three best classifiers, Naive Bayes, QDA, and AdaBoost
### License

This dataset was taken from UCI repository




