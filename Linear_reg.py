#Linear_Regression Using Oops
#---------------------------------#


import numpy as np
import math

class Linear_Regression:
    def Get(self):
        self.bill_amount=int(input("Enter Bill Amount :"))
        self.meal_amount=np.array([34,108,64,88,99,51])
        self.tip_amount=np.array([5,17,11,8,14,5])
    def Show(self):
        self.mean_tb=np.mean(self.meal_amount)
        self.mean_ta=np.mean(self.tip_amount)

        self.bd=np.array(self.meal_amount-self.mean_tb)
        self.td=np.array(self.tip_amount-self.mean_ta)

        self.dp=self.bd*self.td

        self.sqar_bds=self.bd*self.bd

        self.mean_dp=np.sum(self.dp)

        self.mean_sqar_bds=np.sum(self.sqar_bds)

        self.b1=self.mean_dp/self.mean_sqar_bds

        self.b0=self.mean_ta-self.b1*self.mean_tb


        self.yi=(self.b0+self.b1*self.bill_amount)
        print('Predict Amount:',self.yi)
    
    def Calculate_Error(self):
    
        if self.bill_amount in self.meal_amount:
            actual_tip = self.tip_amount[np.where(self.meal_amount == self.bill_amount)[0][0]]
           
            error = abs(self.yi - actual_tip)

            # Print the actual tip and error
            print(f"Actual Tip Amount: {actual_tip}")
            print(f"Prediction Error: {error:.2f}")
        else:
            print("The entered bill amount is not in the dataset.")
            
model=Linear_Regression()
model.Get()
model.Show()
model.Calculate_Error()