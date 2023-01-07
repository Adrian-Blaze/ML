import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn import linear_model, model_selection
df = pd.read_csv(f'C:/Users/user/Desktop/HR.csv')
print(df.head())
##japa = df[df.left==1]
#print(pd.crosstab(df.Department,df.left))
a = pd.get_dummies(df.salary)
b = pd.concat([df, a], axis='columns')
b = b.drop(['last_evaluation', 'number_project', 'time_spend_company', 'Work_accident', 'left', 'Department', 'salary', 'medium'], axis='columns')
print(b.head())
x1,x2,x3,x4 = model_selection.train_test_split(b, df.left, test_size=0.1)
reg = linear_model.LogisticRegression()
reg.fit(x1, x3) 
print(reg.predict(x2))
print(reg.score(x2, x4))

