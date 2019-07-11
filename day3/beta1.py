import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('headbrain.csv')
x=data.iloc[:,0:1].values # reading the values
y=data.iloc[:,1].values
xmean=x.mean()
ymean=y.mean()
print(xmean)
print(ymean)
a=x-xmean
b=y-ymean
c=(x-xmean)**2
for i in range(0,len(x)):  # for loop for summation of beta value (lenth of x and y are same)
   beta=(np.sum(a*b))/np.sum(c)
print('beta value=',beta) 
beta1=ymean-beta*xmean #B1=ymean-B*xmean
print(beta1)  
   
# finding performance
