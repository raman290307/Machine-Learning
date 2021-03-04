#!/usr/bin/env python
# coding: utf-8

# In[2]:


#predicting selection of the student
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data=pd.read_csv('C:\\Users\\Karan\\Downloads\\0_Student_selection_dataset.csv')
print(data)


# In[3]:


x=data.iloc[:,:-1]
y=data.iloc[:,-1:]
print(x)
print(y)


# In[4]:


model = LinearRegression()
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)
print(x_train)
print(y_train)


# In[7]:


from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
x_train =sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
print(x_train)
print(x_test)


# In[8]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
pred = model.predict(x_test)
print(pred)
print(model.intercept_)
print(model.coef_)


# In[9]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
print(y_pred)


# In[40]:


#computing confusion matrix
from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,y_pred)
print('confusion_matrix: \n',c)
p= classifier.predict_proba(x_test)[:,:]
print(p)


# In[10]:


#efficiency of data
from sklearn.metrics import r2_score
score= r2_score(y_test,y_pred)
print(score)


# In[11]:


#predicting slection with 90%, 5 yrs experience, 8 test score and 10 interview score 
x=model.predict([[5,8,10,90]])
print(x)


# In[12]:


#predicting slection with 75%, 8 yrs experience, 7 test score and 6 interview score 
y=model.predict([[8,7,6,75]])
print(y)

