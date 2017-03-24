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

### Classifiers used to predict the breast cancer for a training size = 300 

Type of Classifiers | Training Time | Prediction Time| F1 score (training set) | **F1 Score (testing set)**
:---:|:---:|:---:|:---:|:---:
**K-nearest neighbor** | .0012 | .0022 | .9264 | .9081
**Decision Trees** | .0033 | .0002 | 1.000 | .9239
**SVC** | .0051 |.0029 | 1.000 |.0000
**Naive Bayes** |.0006 | .0003 |.9058 | .9341
**Random Forest** | .0381 | .0064 | 1.000 | .9101
**AdaBoost** | .1866 | .0055 | 1.000 | .9451
**QDA** |.0013 | .0005 |



### License

This dataset was taken from UCI repository
