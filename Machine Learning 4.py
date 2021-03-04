#!/usr/bin/env python
# coding: utf-8

# In[16]:


#creating svm classifier model
import pandas as pd
import numpy as np
data=pd.read_csv('C:\\Users\\Karan\\Downloads\\breast_cancer_data.csv')
print(data)
print(data.shape)


# In[18]:


d1=pd.get_dummies(data['diagnosis'],drop_first=True)
print(d1)
data1=pd.concat([d1,data],axis=1)
print(data1)


# In[19]:


data2=data1.drop('diagnosis',axis=1)
print(data2)


# In[20]:


x=data1.iloc[:,2:32].values
y=data1.iloc[:,0:1].values
print(x.shape)
print(y.shape)


# In[21]:


data1.isnull().sum()#checking null value


# In[22]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=1)
print(len(x_train))
print(len(x_test))


# In[23]:


#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
print(x_train)
print(x_test)


# In[24]:


#fit the svm model to traing dataset
from sklearn.svm import SVC
svm_model=SVC(kernel='rbf')
svm_model.fit(x_train,y_train)


# In[25]:


#prediction on test dataset
y_pred=svm_model.predict(x_test)
print(y_pred)


# In[26]:


#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


# In[27]:


#checking accuracy of the model
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print(acc)

