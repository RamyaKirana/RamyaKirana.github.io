import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
data=pd.read_csv('Salary_Data.csv')
x=data.iloc[:,0:1].values
y=data.iloc[:,1].values

plt.scatter(x,y,color='red')
