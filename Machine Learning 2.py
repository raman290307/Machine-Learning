#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
data=pd.read_csv('C:\\Users\\Karan\\Downloads\\_salary_predict_dataset1.csv')
print(data)


# In[17]:


data2=data.drop('Salary',axis=1)
print(data2)


# In[18]:


x=data.iloc[:,:-1]
y=data.iloc[:,-1:]
print(x)
print(y)


# In[19]:


from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)
print(x_train)
print(y_train)


# In[22]:


model=LinearRegression()
model.fit(x_train,y_train)
print(model.coef_)
print(model.intercept_)


# In[23]:


pred= model.predict(x_test)
print(pred)


# In[11]:


from sklearn.metrics import r2_score
score = r2_score(y_test,pred)
print(score)


# In[12]:


#salary of candidate with 5 yrs experience,8 test score and 10 interview score
x=model.predict([[5,8,10]])
print(x)


# In[13]:


#salary of candidate with 8 yrs experience,7 test score and 6 interview score
y=model.predict([[8,7,6]])
print(y)

