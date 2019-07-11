import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 #graph predict
data=pd.read_csv('Salary_Data.csv')
x=data.iloc[:,0:1].values
y=data.iloc[:,1].values
#print(x)
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


#predict the salary 
a=input('Enter the exp sep by comma')
a=a.split(',')
result=[]
for x in a:
     result.append(float(x))
result    
result=np.array(result)
result=result.reshape((len(result),1))
regressor.predict(result)

#linear regression
#from sklearn.matrics import mean_squared_error
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
rmse
#calculating squares
sample=0
for i in range(0,len(y_test)):
    sample +=(y_test[i]- y_pred[i])**2
    
res=sample/len(y_test)
import math
math.sqrt(res)
   
   

 