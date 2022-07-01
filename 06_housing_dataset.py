from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

bos = load_boston
reg = LinearRegression()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('HousingData.csv')

x = df.drop('CRIM',axis=1)
x = x.drop('ZN',axis=1)
x = x.drop('CHAS',axis=1)
x = x.drop('PTRATIO',axis=1)
x = x.drop('MEDV',axis=1)
y = df['MEDV']

x['INDUS'].fillna((x['INDUS'].mean()), inplace=True)
x['AGE'].fillna((x['AGE'].mean()), inplace=True)
x['LSTAT'].fillna((x['LSTAT'].mean()), inplace=True)

# print(x.info())

print(y)
print(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0,test_size=0.3)

reg_train = reg.fit(x_train,y_train)
reg_pred = reg.predict(x_test)

print(mean_squared_error(y_test,reg_pred))