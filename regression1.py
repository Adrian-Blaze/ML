import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from  sklearn import linear_model

df = pd.read_csv(f'C:/Users/user/Desktop/homeprices1.csv')
print(df)
B = math.floor(df.bedrooms.median())
print('median is', B)
df.bedrooms = df.bedrooms.fillna(B)
print(df)
reg2 = linear_model.LinearRegression()
reg2.fit(df[['area', 'bedrooms', 'age']], df.price)
print(reg2.predict([[3000,3,40]]))
import pickle
with open('reg2', 'wb') as f:
    pickle.dump(reg2, f)
with open('reg2', 'rb') as f:
    r2 = pickle.load (f)
print(r2.predict([[3000,3,40]]))





