#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
df=pd.read_csv('Redwine.csv')
df.head(10)


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


sns.heatmap(df.isnull())  #checking heatmap for null values
plt.title("Null values")
plt.show()


# In[8]:


df["quality"].hist(grid=False)
plt.title("Red Wine Quality")
plt.show()


# In[9]:


df.columns


# # checking outlier

# In[13]:


df.plot(kind='box',subplots=True,layout=(2,6),figsize=(20,10))


# # checking skewness

# In[14]:


df.skew()


# # Removing Outlier using z score

# In[15]:


from scipy.stats import zscore
import numpy as np
z=np.abs(zscore(df))# abs will make it positive---> mod of x= x and -X---> if x=2 then mode of x=3 or x=-(-2)=2
z.shape


# In[16]:


thresold=3
print(np.where(z>3))


# In[17]:


df_new=df[(z<3).all(axis=1)]


# In[20]:


df_new


# In[19]:


df_new.plot(kind='box',subplots=True,layout=(2,6),figsize=(20,10))


# # Checking correlation

# In[23]:


corr_mat=df_new.corr()#----------------> corelation function

plt.figure(figsize=[22,12])#-------> figure size dimetions
sns.heatmap(corr_mat,annot=True)#--> annot = true means will print values
plt.title("correlation Matrix")#---> title
plt.show()


# In[24]:


corr_matrix=df_new.corr()
corr_matrix["quality"].sort_values(ascending = False) #----> corelation of all columns with sex


# In[25]:


df_new.skew()


# # Splitting independent and target variable in x and Y before removing skewness

# In[26]:


x=df_new.drop("quality",axis=1)
y=df_new["quality"]


# In[27]:


x.skew()


# # power transform function : Removing Skweness

# In[28]:


from sklearn.preprocessing import power_transform
x_new=power_transform(x)

x=pd.DataFrame(x_new,columns=x.columns)


# In[29]:


x.skew()


# # Applying Decision Tree Algorithem as suggested

# In[32]:


x.shape


# In[33]:


y.shape


# In[42]:


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


# In[48]:


import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')



#x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=.33,random_state=42)
model=[DecisionTreeClassifier()]#[DecisionTreeClassifier(),SVC(),KNeighborsClassifier(),MultinomialNB()]
for m in model:
    m.fit(x_train,y_train)
    m.score(x_train,y_train)
    predm=m.predict(x_test)
    print("Accuracy",accuracy_score(y_test,predm)*100)
    print(confusion_matrix(y_test,predm)*100)
    print(classification_report(y_test,predm)*100)
    



# # Cross Validation

# In[50]:


from sklearn.model_selection import cross_val_score
scr=cross_val_score(dt,x,y,cv=5)
print("Result for dt:",scr.mean())


# # Hyper Parameter Tunning

# In[89]:


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


# In[90]:


GCV=GridSearchCV(RandomForestClassifier(),parameters,cv=5,scoring="accuracy")
GCV.fit(x_train,y_train)
GCV.best_params_


# In[91]:


GCV_pred=GCV.best_estimator_.predict(x_test)
accuracy_score(y_test,GCV_pred)


# # ROC AUC Plot

# In[92]:


from sklearn.metrics import plot_roc_curve
plot_roc_curve(GCV.best_estimator_,x_test,y_test)
plt.title("ROC AUC plot")
plt.show()


# In[ ]:




