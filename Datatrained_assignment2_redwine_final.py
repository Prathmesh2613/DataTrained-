#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[3]:


import pandas as pd
df=pd.read_csv('Redwine.csv')
df.head(10)


# In[4]:


df.shape


# In[5]:


df.describe()


# In[8]:


df.info()


# In[9]:


sns.heatmap(df.isnull())  #checking heatmap for null values
plt.title("Null values")
plt.show()


# In[10]:


df["quality"].hist(grid=False)
plt.title("Red Wine Quality")
plt.show()


# # checking outlier

# In[11]:


df.plot(kind='box',subplots=True,layout=(2,6),figsize=(20,10))


# # checking skewness

# In[12]:


df.skew()


# # Checking correlation

# In[14]:


corr_mat=df.corr()#----------------> corelation function

plt.figure(figsize=[22,12])#-------> figure size dimetions
sns.heatmap(corr_mat,annot=True)#--> annot = true means will print values
plt.title("correlation Matrix")#---> title
plt.show()


# In[16]:


df['quality'].value_counts()


# # Splitting independent and target variable in x and Y before removing skewness

# In[18]:


x=df.drop("quality",axis=1)


# In[19]:


print(x)


# In[21]:


y=df['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)


# In[22]:


print(y)


# # Applying Decision Tree Algorithem as suggested

# In[23]:


import sklearn
from sklearn.tree import DecisionTreeClassifier
#from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=.33,random_state=42)
dt=DecisionTreeClassifier()
#dt=MultinomialNB()
dt.fit(x_train,y_train)
predrf=dt.predict(x_test)
acc=accuracy_score(y_test,predrf)

print("Accuracy",accuracy_score(y_test,predrf)*100)
print(confusion_matrix(y_test,predrf)*100)
print(classification_report(y_test,predrf)*100)


# # Removing Outlier using z score

# In[49]:


from scipy.stats import zscore
import numpy as np
z=np.abs(zscore(df))# abs will make it positive---> mod of x= x and -X---> if x=2 then mode of x=3 or x=-(-2)=2
z.shape


# In[50]:


thresold=3
print(np.where(z>3))


# In[51]:


df_new=df[(z<3).all(axis=1)]


# In[52]:


df_new


# In[53]:


x=df_new.drop("quality",axis=1)


# In[54]:


print(x)


# In[55]:


y=df_new['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)


# In[56]:


print(y)


# In[57]:


import sklearn
from sklearn.tree import DecisionTreeClassifier
#from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=.33,random_state=42)
dt=DecisionTreeClassifier()
#dt=MultinomialNB()
dt.fit(x_train,y_train)
predrf=dt.predict(x_test)
acc=accuracy_score(y_test,predrf)

print("Accuracy",accuracy_score(y_test,predrf)*100)
print(confusion_matrix(y_test,predrf)*100)
print(classification_report(y_test,predrf)*100)


# # Cross Validation

# In[58]:


from sklearn.model_selection import cross_val_score
scr=cross_val_score(dt,x,y,cv=5)
print("Result for dt:",scr.mean())


# # Hyper Parameter Tunning

# In[59]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
#from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

parameters={'max_features':['auto','sqrt','log2'],
            'max_depth': [4,5,6,7,8],
            'criterion':['gini','entropy']}


# In[60]:


GCV=GridSearchCV(RandomForestClassifier(),parameters,cv=5,scoring="accuracy")
GCV.fit(x_train,y_train)
GCV.best_params_


# In[61]:


GCV_pred=GCV.best_estimator_.predict(x_test)
accuracy_score(y_test,GCV_pred)


# # ROC AUC Plot

# In[63]:


from sklearn.metrics import plot_roc_curve
plot_roc_curve(GCV.best_estimator_,x_test,y_test)
plt.title("ROC AUC plot")
plt.show()


# # saving the model in pickel format

# In[64]:


import joblib
joblib.dump(GCV.best_estimator_,"Redwine.pk1")


# In[ ]:




