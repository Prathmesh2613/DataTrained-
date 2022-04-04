#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')


# In[99]:


df=pd.read_csv('salary.csv')
df.head(10)


# In[5]:


df.describe()


# In[14]:


df.discipline.unique()


# In[12]:


df.columns


# # Data conversion

# In[100]:


lencode=preprocessing.LabelEncoder()
df['sex']=lencode.fit_transform(df['sex'])


# In[101]:


df.sex.unique()


# In[102]:


lencode=preprocessing.LabelEncoder()
df['rank']=lencode.fit_transform(df['rank'])


# In[103]:


df.head()


# In[104]:


lencode=preprocessing.LabelEncoder()
df['discipline']=lencode.fit_transform(df['discipline'])


# In[105]:


df.discipline.unique()


# In[106]:


df.head()


# # checking null values

# In[27]:


df.isnull().sum()


# # checking skewness 

# In[28]:


df.skew()


# # checking correlation

# In[107]:


corr_mat=df.corr()#----------------> corelation function

plt.figure(figsize=[22,12])#-------> figure size dimetions
sns.heatmap(corr_mat,annot=True,cmap="Blues")#--> annot = true means will print values
plt.title("correlation Matrix")#---> title
plt.show()


# In[31]:


plt.plot(df['salary'],df['rank'])


# In[32]:


plt.plot(df['salary'],df['discipline'])


# In[33]:


plt.plot(df['salary'],df['yrs.since.phd'])


# # Splitting independent and target variable in x and Y before removing skewness and outliers

# In[113]:


x=df.drop("salary",axis=1)


# In[114]:


y=df['salary']


# In[115]:


x


# In[83]:


from sklearn.model_selection import train_test_split


# In[84]:


x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=.20,random_state=42)


# In[111]:


import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=.33,random_state=42)
dt=LogisticRegression()
#dt=MultinomialNB()
dt.fit(x_train,y_train)
predrf=dt.predict(x_test)
acc=accuracy_score(y_test,predrf)

print("Accuracy",accuracy_score(y_test,predrf)*100)
print(confusion_matrix(y_test,predrf)*100)
print(classification_report(y_test,predrf)*100)


# In[ ]:




