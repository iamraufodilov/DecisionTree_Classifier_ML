# import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#loading data
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(r"G:\rauf\STEPBYSTEP\Data\pima-indians-diabetes.csv", header=None, names=col_names)
print(pima.head)

# split data
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols]
y = pima.label
print(X[:5], y[:5])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# create model
DTClf = DecisionTreeClassifier()

# train model
DTClf.fit(X_train, y_train)

# make predict
y_pred = DTClf.predict(X_test)

# get classification report
my_confusion = confusion_matrix(y_test, y_pred)
my_accuracy = accuracy_score(y_test, y_pred)
my_report = classification_report(y_test, y_pred)

print("matrix result from model is: \n", my_confusion)
print("accuracy score from model is: \n", my_accuracy)
print("classification report from model is: \n", my_report)