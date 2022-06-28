# import pandas as pd
# df = pd.read_csv("IRIS.csv")
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
# logr = LogisticRegression()
# lr = LogisticRegression(random_state=0)
#
# X = df.drop("species",axis=1)
# y = df["species"]
#
# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.3)
#
# train = logr.fit(X_train,y_train)
# y_pred = logr.predict(X_test)
#
# print(accuracy_score(y_test,y_pred))

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

rf = RandomForestClassifier(random_state=1)
lr = LogisticRegression(random_state=0)
gbm = GradientBoostingClassifier(n_estimators=10)
dt = DecisionTreeClassifier(random_state=0)
sv = svm.SVC()
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2),random_state=0)
nb=MultinomialNB()

df = pd.read_csv("IRIS.csv")

X = df.drop("species",axis=1)
y = df["species"]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.3)

rf_train = rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)

lr_train = lr.fit(X_train,y_train)
lr_pred = lr.predict(X_test)

dt_train = dt.fit(X_train,y_train)
dt_pred = dt.predict(X_test)

nb_train = nb.fit(X_train,y_train)
nb_pred = nb.predict(X_test)

print('random forest')
print(accuracy_score(y_test,rf_pred))

print('logistic')
print(accuracy_score(y_test,lr_pred))

print('decision tree')
print(accuracy_score(y_test,dt_pred))

print('naive bayes')
print(accuracy_score(y_test,nb_pred))