import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

df1 = pd.read_csv("walmart/train.csv")
reg = LinearRegression()
print(df1)

x = df1.drop('Weekly_Sales',axis=1)
y = df1['Weekly_Sales']

le = preprocessing.LabelEncoder()
df1['Date']= le.fit_transform(df1['Date'])
x = df1

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.2)

reg_train = reg.fit(x_train,y_train)
reg_pred = reg.predict(x_test)

print(mean_squared_error(y_test,reg_pred))