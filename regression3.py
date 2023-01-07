import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn import linear_model
obj = {'area': [2600, 3000, 3200, 3600, 4000], 'price' : [550000, 565000, 610000, 680000, 725000]}
df = pd.DataFrame(obj)
print(df)
#%matplotlib inline
plt.scatter(df.area, df.price, marker='+')
plt.xlabel('area')
plt.ylabel('price')
plt.show()
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)
#print('Good')
print(reg.predict([[3300]]))
plt.plot(df.area, reg.predict(df[['area']]), color='blue')
plt.show()
