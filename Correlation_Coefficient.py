import numpy as np
import pandas as pd
import math

class correlation_coefficient:
    def Get(self):
        self.meal_amount=np.array([34,108,64,88,99,51])
        self.tip_amount=np.array([5,17,11,8,14,5])

    def Show(self):
        self.Sum_multiply_xy=np.sum(self.meal_amount*self.tip_amount)
        self.sum_x=np.sum(self.meal_amount)
        self.sum_y=np.sum(self.tip_amount)
        self.n=len(self.meal_amount)
        self.squar_sum_x=np.sum(self.meal_amount*self.meal_amount)
        self.squar_sum_y=np.sum(self.tip_amount*self.tip_amount)

        self.R=((self.n*self.Sum_multiply_xy)-(self.sum_x*self.sum_y))

        self.cx=(self.n*self.squar_sum_x)-(self.sum_x*self.sum_x)
        self.cy=(self.n*self.squar_sum_y)-(self.sum_y*self.sum_y)

        self.D=math.sqrt(self.cx*self.cy)

        self.cr=self.R/self.D
        print('correlation coefficient :',f'{self.cr:.1f}')
ir=correlation_coefficient()
ir.Get()
ir.Show()
