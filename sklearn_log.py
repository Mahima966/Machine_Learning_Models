import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import math


df=pd.read_csv('C:\ML_folder\CreditScore.csv')
y=df['Approved']
print(y)
x=df['CreditScore']

x=np.array(x)
y=np.array(y)
  
x=x.reshape(len(x),1)

model=LogisticRegression()
model.fit(x,y)
print("intercept",model.intercept_)
print("slope",model.coef_)
j=model.score(x,y)
print("score:",j)

x1=float(input("Enter Your Credit Score:"))

user_input=np.array([[x1]])
p=model.predict(user_input)
if p[0]==1:
      print("approved")
else:
    print("not Approved!")


