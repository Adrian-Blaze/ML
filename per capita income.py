import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn import linear_model
df = pd.read_csv(f'C:/Users/user/Desktop/canada_PCI.csv')
print(df)
reg = linear_model.LinearRegression()
reg.fit(df[['year']], df.PCI)
plt.scatter(df.year, df.PCI, marker='+')
plt.xlabel('year')
plt.ylabel('per capita income  (US)')
plt.plot(df.year, reg.predict(df[['year']]), color='blue')
plt.show()
print(reg.predict([[2020]]))