import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn import linear_model
df = pd.read_csv(f'C:/Users/user/Desktop/carprices.csv')
#print(df)
df1 = pd.get_dummies(df.Car_Model)
df2 = pd.concat([df, df1], axis='columns')
#print(df2)
df3 = df2.drop(['Car_Model', 'Audi A5'], axis='columns')
#print(df3)
X = df3.drop(['Sell_Price'], axis='columns')
print(X)
reg = linear_model.LinearRegression()
reg.fit(X, df.Sell_Price)
print(reg.predict([[0,0,0,1]]))

