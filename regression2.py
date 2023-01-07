import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn import linear_model
from word2number import w2n
df = pd.read_csv(f'C:/Users/user/Desktop/hiring.csv')
df.experience = df.experience.fillna(0)
#print (df.test_score)
B = df.test_score.median()
df.test_score = df.test_score.fillna(B)
#print(df.experience)
i = 2
while i < 8:
    df.experience[i] = w2n.word_to_num(df.experience[i])
    i = i + 1
print(df)
reg = linear_model.LinearRegression()
reg.fit(df[['experience', 'test_score', 'interview_score']], df.salary)
print(reg.predict([[12,10,10]]))


