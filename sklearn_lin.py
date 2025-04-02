import numpy as np
import pandas as pd    
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

df=pd.read_csv("c:\ML_folder\MealFile.csv")
y=df['TipAmount']
print(y)
x=df['TotalBill']

x=np.array(x)
y=np.array(y)

p=np.array([400,200,56,78,77,88])  #our pridictions
p=p.reshape(len(p),1)    #for multiple dimension we'll reshape it
x=x.reshape(len(x),1)
y=y.reshape(len(y),1)


model=LinearRegression()
model.fit(x,y)
print("intercept",model.intercept_)
print("slope",model.coef_)
j=model.score(x,y)
print("score:",j)
c=math.sqrt(j)
print("sqrt Score:",c)
print(model.predict(p))


yp=model.predict(p)
print("predict\n",yp)
lse=mean_squared_error(y,yp)
diff=y-yp
print("diff\n",diff)
lse=np.mean(diff**2)
print(lse)