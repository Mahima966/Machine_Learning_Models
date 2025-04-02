#logistics Regression Using Oops
#---------------------------------#

import math
import numpy as np

class Logistics_Regression:
    def Get(self):
        self.x1=int(input("Enter your credit :"))
        self.credit_score=np.array([655, 692,  681, 663, 688, 693, 699, 699, 683, 698, 655, 703, 704, 745, 702])
        self.approval=np.array([0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1])

    def Show(self):
        self.mean_credit_score=np.mean(self.credit_score)
        self.mean_approval=np.mean(self.approval)

        # Differences from mean
        self.bd=np.array(self.credit_score-self.mean_credit_score)
        self.td=np.array(self.approval-self.mean_approval)

        # Dot products  
        self.dp=self.bd*self.td

        # squares
        self.sqar_bds=self.bd*self.bd

        # sum of mean value 
        self.mean_dp=np.sum(self.dp)
        self.mean_sqar_bds=np.sum(self.sqar_bds)

        self.b1=self.mean_dp/self.mean_sqar_bds # Slope

        self.b0=self.mean_approval-self.b1*self.mean_credit_score # Intercept

        # Prediction for a new value
        p=math.exp(self.b0+self.b1*self.x1)/(1+math.exp(self.b0+self.b1*self.x1))
        print("predicted_probability :",f'{p:.0f}')

        print("Approved") if 1>=p else print("Not Approved")

model=Logistics_Regression()
model.Get()
model.Show()
