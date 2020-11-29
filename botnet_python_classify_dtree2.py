import scipy as sp
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

data = sp.genfromtxt("C:\\Botnet\\dataset\\NugacheBot1Normal-data.csv", delimiter=",")
X1 = data[:,0]
X2 = data[:,1]
target = sp.genfromtxt("C:\\Botnet\\dataset\\NugacheBot1Normal-target.csv", delimiter=",")

X_train, X_test, y_train, y_test = tts(data, target, test_size=0.25, random_state=42)
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print("Classification Report:")
print(metrics.classification_report(y_test, predicted))
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, predicted))
print("Accuracy Score :")
print(metrics.accuracy_score(y_test, predicted))

print('We are done with Decision Tree')
    

