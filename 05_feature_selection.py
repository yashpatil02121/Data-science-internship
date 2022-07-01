# import pandas as pd
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
#
# df = pd.read_csv("IRIS.csv")
#
# X = df.drop("species",axis=1)
# Y = df["species"]
#
# bestfeature = SelectKBest(score_func=chi2 ,k='all')
# fit = bestfeature.fit(X,Y)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X.columns)
# featuresScore = pd.concat([dfcolumns, dfscores], axis=1)
# featuresScore.columns = ['specs', 'score']
#
# print(featuresScore)
#
#
# # feature selection 2
# from sklearn.ensemble import ExtraTreesClassifier
# import matplotlib.pyplot as plt
#
# model = ExtraTreesClassifier()
# model.fit(X,Y)
# print(model.feature_importances_)
#
# feat_importance = pd.Series(model.feature_importances_, index = X.columns)
# feat_importance.nlargest(4).plot(kind='barh')
# plt.show()


# Dataset conversion - numerical to categorial
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("IRIS.csv")

rf = RandomForestClassifier()
df['sepal_length'] = pd.cut(df['sepal_length'],3, labels = ['1','2','3'])
df['sepal_width'] = pd.cut(df['sepal_width'],3, labels = ['1','2','3'])
df['PetalLengthCm'] = pd.cut(df['PetalLengthCm'],3, labels = ['1','2','3'])
df['PetalWidthCm'] = pd.cut(df['PetalWidthCm'],3, labels = ['1','2','3'])

X = df.drop('Id', axis=1)
X = X.drop('Species', axis=1)
Y = df['Species']
print(Y)
le = LabelEncoder()
le.fit(Y)
Y = le.transform(Y)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=0, test_size= 0.3)

rf.fit(X_train, Y_train)
y_pred = rf.predict(X_test)
print('Random Forest: ', accuracy_score(Y_test, y_pred))
