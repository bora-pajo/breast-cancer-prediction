## Predicting malignant versus benign breast cancer cases
Predicting the probability that a diagnosed breast cancer case is malignant or benign based on Wisconsin dataset from UCI repository. 

#### Files in this repository
Many files in the respository are examples and things to be saved as the work progresses. Those are names examples_01 and so on. 
The actual best and cleanest file is called breast_cancer_clean.ipynb 


### Libraries used
numpy, pandas, matplotlib.pyplot, sklearn.linear_model, sklearn.pipeline, sklearn.preprocessing, sklearn.neighbors, sklearn.naive_bayes, scipy

### Uses
This project tests various classifiers (logistic regression, SVM, decision trees, random forest, and others) to find the classifier that predicts the testing data with the best possible accuracy at the shortest time possible.  

First, it examines the data looking for patterns of correlations among variables. Then it runs various classifiers and compare them. Once a number of 2-3 classifiers is identified as most accurate, the model is fine tuned again. Then, we can see which model predicts data with the highest accuracy.  


### Graphs
Some examples from bokeh and lightning as well as matplolib plots.

### Classifiers used to predict the breast cancer
|**Type of Classifiers**| **Training Set Size** | Training Time | Prediction Time | F1 Score (train) | **F1 score (test)** |
|----------------------:|:--------------------: |:-------------:|:---------------:|:----------------:|:-------------------:|
| K-nearest neighbor    |         100           |     .0018     |      .0019      |     .8857        |         .8588       |
|                       |                       |               |                 |                  |                     |


Type of Classifiers | Training Size | Training Time | Prediction Time| F1 score (training set) | **F1 Score (testing set)**
:---:|:---:|:---:|:---:|:---:|:---:
**K-nearest neighbor** | 100 | .0018 | .0019 | .8857 | .8588 



### License

This dataset was taken from UCI repository
