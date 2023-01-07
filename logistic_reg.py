import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn import linear_model, model_selection
df = pd.read_csv(f'C:/Users/user/Desktop/insurance_data.csv')
#print(df.head())
plt.scatter(df.age, df.bought_insurance, marker='*')
plt.show()
x1,x2,y1,y2 = model_selection.train_test_split(df[['age']], df.bought_insurance, test_size=0.1)
reg = linear_model.LogisticRegression()
reg.fit(x1,y1)
print(x2)
print(reg.predict(x2))

print(reg.score(x2, y2))

