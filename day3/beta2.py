 import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('headbrain.csv')
x=data.iloc[:,0:1].values # reading the values
y=data.iloc[:,1].values
xmean=x.mean()
ymean=y.mean()
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
b=y-ymean
y_pred=regressor.predict(y)

for i in range(0,len(y)):
    sst=np.sum(b)
for j in range(0,len(y)):
         