## Predicting malignant versus benign breast cancer cases
Predicting the probability that a diagnosed breast cancer case is malignant or benign based on Wisconsin dataset from UCI repository. 

### Project Overview
According to the Centers for Disease Control and Prevention (CDC) breast cancer is the most common type of cancer for women regardless of race and ethnicity (CDC, 2016). Around 220,000 women are diagnosed with breast cancer each year in the United States (CDC, 2016). Although we may not be aware of all the factors contributing in developing breast cancer, certain attributes such as family history, age, obesity, alcohol and tobacco use have been identified from research studies on this topic (DeSantis, Ma, Bryan, & Jemal, 2014). Breast images procedures such as mammography have been found to be quite effective in early identification cases of breast cancer (Ball, 2012).  When breast images procedures are not utilized, patients can find out late about their diagnosis to be able to treat it.  Similar work on attempting to find the best way to predict the type of cancer based on images of mammograms has identified Support Vector Machine as the best predictor after tuning parameters.  
 

#### Files in this repository
Many files in the respository are examples and things to be saved as the work progresses. Those are names examples_01 and so on. 
The file where you can find all the coding for different classifiers is called _breast_cancer_clean.ipynb_ 

##### Problem Statement
This project focuses in investigating the probability of predicting the type of breast cancer (malignant or benign) from the given characteristics of breast mass computed from digitized images.  The cases provided, are cases diagnosed with some type of tumor, but only some of them (approximately 37%) are malignant.  This project will examine the data available and attempt to predict the possibility that a breast cancer diagnosis is malignant or benign based on the attributes collected from the breast mass. To achieve this goal, the following steps are identified:
•	Download the breast cancer images data from UCI repository
•	Familiarize with the data by looking at its shape, the relations between variables, their possible correlations, and other attributes of the dataset. 
•	Preprocess data if needed
•	Split the data into testing and training samples
•	Employ various classifiers (K-neighbors, Decision trees, SVC, QDA, AdaBoost, Naïve Bayes, Random Forest, and MLP classifier) to predict the data with different sets of training samples (100, 200, 300, and 400). 
•	Once the best predicting model is identified, will reduce the training set in size to see what is the limit for this classifier to best predict these data.
•	Compare the best identified classifier with evaluation metric stated at the beginning of the project.
•	Write conclusions. 


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


#### Dataset and Inputs
The characteristics of the cell nuclei have been captured in the images and a classification methods which uses linear programming to construct a decision line. The dataset is published by Kaggle and taken from the University of California Irvine (UCI) machine learning repository.  The data is taken from the Breast Cancer Wisconsin Center. It includes ten (10) attributes taken from each cell nucleus as well as ID and the diagnosis (M=malignant, B=benign).  The dataset has 570 cases and 31 variables.   

#### Evaluation metric

F1 score is a measure of accuracy or the ratio of the data that was accurately predicted. The closer the F1-score is to a value of 1, the best the prediction is, and the closer to a value of 0 it is, the worse the prediction. F1-score considers the true positives and the true negatives, and is best used when comparing various classifiers as I am proposing to do in this dataset.  From the literature, I reached the conclusion that F1-score is the best evaluation metric to be used for this type of classification problem.  
•	The formula for the F1 score from the sklearn documentation is F1 = 2* (precision * recall) / (precision + recall). 
•	Precision is a measure for result relevancy (according to sklearn documentation). It is often referred to as specificity and shows the number of cases that are relevant.  For example, in this particular dataset the number of cases that are identified as malignant would be very relevant. The formula for precision is: P = Tp/(Tp+Fp) where Tp are the true positives (the number of cases that were identified correctly as true (in this case these will be the malignant cases), and Fp are false positives (or the number of cases that are not malignant but are incorrectly identified as such). 
•	Recall on the other hand is considered a measure of sensitivity and is measured as the number of True positives over the number of True positives plus the False negatives. R = Tp/(Tp+Fn)
•	True positives are the correctly identified cases as 1 or malignant in our case
•	True negatives are the correctly identified cases as 0 or benign in our case
•	False positives are the cases that are identified as positive (1 or malignant in this case) but are in fact negative (0 or benign) 
•	False negatives are the cases that are identified as negative (0 or benign in this case) and are in fact positive (1 or malignant). These are the most dangerous in this dataset. 
•	Receiver operating characteristics (ROC) is another measurement metric used in this project.  It is a measurement of the classifier quality based on the true positives and the true negatives. ROC measured the proportion between true positives and true negatives.
•	Area under the curve (AUC) is the last measurement metric used in this project.  It basically shows how well the classifier is performing.  If AUC is closer to a value of 1, the classifier is doing pretty well. If AUC is closer to 0 it is not doing a good job.  


### Graphs
Some examples from bokeh, matplolib plots, prettyplots, and ggplot for python

#### Implementation
First, the dataset was split into training and testing sets randomly. Training set included 400 cases and testing set included 169 cases. The rationale for this division was to see how many cases were the optimal number for training the data with each different classifier and how long it took in each case.  To implement different classifiers, I used training sets of 100, 200, 300, and 400 and got the results of training time, F-score for training, predicting time for the test data and the F-score for the testing set based on each case. The classifiers utilized were: 1) K-nearest neighbor, 2) Decision trees, 3) SVC, 4) Naïve Bayes, 5) Random Forest, 6) AdaBoost, 7) QDA, and 8) MLP. I ran these classifiers two times: 


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

##### ROC and AUC
There are graphs for ROC and AUC provided for each classifier that was used.

#### Improvements
I think there are possible improvements to be done to the other classifiers that were used to predict this data.  Because they did not perform as great as the first three, I dismissed them and continued improving the ones that predicted the best from the very beginning.  This is a common approach humans take on many things, but it is possible to modify the other classifiers that did not perform well initially. Tuning them, or removing highly correlated variables (especially for SVC), could have improved these other classifiers significantly.  

ROC information added for the three best classifiers, Naive Bayes, QDA, and AdaBoost
### License

This dataset was taken from UCI repository




