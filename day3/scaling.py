import numpy as np
a=["delhi","bangalore","chennai","Mumbai"]
from sklearn.preprocessing import LabelEncoder
lEncoder=LabelEncoder()

lEncoder.fit(a)
b=lEncoder.transform(a)
c=["chennai","mumbai"]
lEncoder.fit(c)
lEncoder.transform(c)
lEncoder.inverse_transform([1])

#scaling operation
import numpy as np
import pandas as pd
data=pd.read_csv('50_Startups.csv')
data.columns #i show the name of columns in the read file(50_Startups)
x=data.iloc[:,:4].values
y=data.iloc[:,4].values
#used to set random values in the data set
from sklearn.preprocessing import LabelEncoder
lEncoder=LabelEncoder()
x[:,3]=lEncoder.fit_transform(x[:,3])
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()
#x=x[:,1:] #it wll delete the last colunm it wll give only 0 and 1 col


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
mRegressor=LinearRegression()

mRegressor.fit(x_train,y_train)

y_pred=mRegressor.predict(x_test)
from sklearn.metrics import mean_squared_error
 
mse=mean_squared_error(y_test,y_pred)**(1/2)