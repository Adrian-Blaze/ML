import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn import linear_model
df = pd.read_csv(f'C:/Users/user/Desktop/Python lessons/homeprices.csv')
print(df)
df2  = pd.get_dummies(df.town)
df3 = pd.concat([df, df2], axis='columns')
#print(df3)
df4 = df3.drop(['town', 'west windsor'], axis='columns')
#print(df4)
x = df4.drop(['price'], axis='columns')
model = linear_model.LinearRegression()
#print(x)
model.fit(x, df.price)
print(model.predict([[2000,0,1]]))

#USING LABEL ENCODER

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['town encoded'] = le.fit_transform(df.town)
print(df)
x2 = df.drop(['town','price'], axis='columns')
model = linear_model.LinearRegression()
model.fit(x2, df.price)
print(model.predict([[2000,1]]))



