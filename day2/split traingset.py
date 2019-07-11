import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
data=pd.read_csv('Salary_Data.csv')
x=data.iloc[:,0:1].values
y=data.iloc[:,1].values

#plt.scatter(x,y,color='red')
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=0) #random_state used for same value

from sklearn.linear_model import LinearRegression
regressor=LinearRegression() #create  an instance for class LinaerRegression

regressor.fit(x_train,y_train)#x_train is input and y_train is output
m=regressor.coef_
c=regressor.intercept_

y75=(m*7.5)+c
yP75=regressor.predict([[7.5]])

y_pred=regressor.predict(x_test)

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')

