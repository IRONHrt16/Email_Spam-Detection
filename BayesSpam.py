import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#Step 1: Initialize dataset
df = pd.read_csv("F:/TY_CSE/MachineLearning/Expt/Mini_Project/spam1.csv") # Data-set Location or Address
df.head()

#Step 2: Training data and Test data
df.groupby('Label').describe()
df['spam']=df['Label'].apply(lambda x: 1 if x=='spam' else 0)
print(df.head(10))

X_train, X_test, y_train, y_test = train_test_split(df.EmailText,df.spam)

#Step 3: Feature Extraction
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:3000]

#Step 4: Model building
model = MultinomialNB()
model.fit(X_train_count,y_train)

emails = pd.read_csv("F:/TY_CSE/MachineLearning/Expt/Mini_Project/spam1.csv") # Data-set Location or Address
emails_count = v.transform(emails)
model.predict(emails_count)


X_test_count = v.transform(X_test)
model.score(X_test_count, y_test)

#Step 5: Test Accuracy

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(X_train, y_train)

clf.score(X_test,y_test)
res=(model.score(X_test_count, y_test))*100
print("Accuracy for Naive Bayes Classification algorithm is", res,"%")