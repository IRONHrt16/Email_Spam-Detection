from operator import le
import pandas as pd
from sklearn import *
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import CountVectorizer

#Step 1: Initialize dataset
dataframe = pd.read_csv("F:/TY_CSE/MachineLearning/Expt/Mini_Project/spam1.csv") # Data-set Location or Address
print(dataframe.head(10))

#Step 2: Training data and Test data
x=dataframe["Label"]
y=dataframe["EmailText"]


x_train, y_train= x[:3050], y[:3050]
x_test, y_test = x[3050:], y[3050:]

#Step 3: Feature Extraction
cv = CountVectorizer()
features = cv.fit_transform(x_train)


#Step 4: Model building
model = svm.SVC()
model.fit(features, y_train)
#print(model.best_params_)

#Step 5: Test Accuracy
features_test = cv.transform(x_test)
res=(model.score(features_test, x_test))*100
print("Accuracy for SVM classification algorithm is", res,"%")