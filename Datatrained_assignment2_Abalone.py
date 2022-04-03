#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# In[1]:


import pandas as pd
df=pd.read_csv('Abalone.csv')
df.head(10)


# In[3]:


df.Rings.unique()


# In[5]:


df.Sex.unique()


# In[6]:


df.shape


# # Checking for Null values

# In[7]:


df.isnull().sum()


# # checking for Skewness of Data

# In[8]:


df.skew()


# # Checking for Outliers

# In[9]:


df.describe()


# In[66]:


df.plot(kind='box',subplots=True,layout=(2,6),figsize=(20,10))


# # Checking for Datatypes

# In[10]:


df.dtypes


# In[25]:


from sklearn import preprocessing


# In[26]:


lencode=preprocessing.LabelEncoder()
df['Sex']=lencode.fit_transform(df['Sex'])


# In[27]:


df.Sex.unique()


# # correlation

# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[12]:


corr_mat=df.corr()#----------------> corelation function

plt.figure(figsize=[22,12])#-------> figure size dimetions
sns.heatmap(corr_mat,annot=True,cmap="Blues")#--> annot = true means will print values
plt.title("correlation Matrix")#---> title
plt.show()


# In[13]:


df['Age']=0


# In[14]:


def update_age (row):
    if row['Rings']>=1 and row['Rings']<=8:
        return 1
    elif row['Rings']>=9 and row['Rings']<=10:
        return 2
    elif row['Rings']>=11 and row['Rings']<=29:
        return 3
    return 0


# In[15]:


df['Age']=df.apply(lambda row: update_age(row),axis=1)


# In[16]:


df.dtypes


# In[29]:


df


# In[30]:


corr_mat=df.corr()#----------------> corelation function

plt.figure(figsize=[22,12])#-------> figure size dimetions
sns.heatmap(corr_mat,annot=True,cmap="Blues")#--> annot = true means will print values
plt.title("correlation Matrix")#---> title
plt.show()


# # Splitting independent and target variable in x and Y before removing skewness and outliers

# In[33]:


x=df.drop("Rings",axis=1)


# In[34]:


y=df['Rings']


# In[35]:


x


# In[36]:


y


# In[23]:


df.corr()


# # Applying Decision Tree Algorithem as suggested

# In[56]:


import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=.20,random_state=42)
dt=DecisionTreeRegressor()
#dt=MultinomialNB()
dt.fit(x_train,y_train)
predrf=dt.predict(x_test)
acc=accuracy_score(y_test,predrf)

predicted_test_y = dt.predict(x_test)
predicted_train_y = dt.predict(x_train)

print("Accuracy",accuracy_score(y_test,predrf)*100)
print(confusion_matrix(y_test,predrf)*100)
print(classification_report(y_test,predrf)*100)


# In[51]:


def scatter_y(true_y, predicted_y):
    """Scatter-plot the predicted vs true number of rings
    
    Plots:
       * predicted vs true number of rings
       * perfect agreement line
       * +2/-2 number dotted lines

    Returns the root mean square of the error
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(true_y, predicted_y, '.k')
    
    ax.plot([0, 30], [0, 30], '--k')
    ax.plot([0, 30], [2, 32], ':k')
    ax.plot([2, 32], [0, 30], ':k')
    
    rms = (true_y - predicted_y).std()
    
    ax.text(25, 3,
            "Root Mean Square Error = %.2g" % rms,
            ha='right', va='bottom')

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    
    ax.set_xlabel('True number of rings')
    ax.set_ylabel('Predicted number of rings')
    
    return rms


# In[57]:


scatter_y(y_train, predicted_train_y)
plt.title("Training data")
scatter_y(y_test, predicted_test_y)
plt.title("Test data");


# # Plot learning curves

# In[60]:


data_percentage_array = np.linspace(10, 100, 10)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


train_error = []
test_error = []
for data_percentage in data_percentage_array:
    model = DecisionTreeRegressor(max_depth=10)
    number_of_samples = int(data_percentage / 100. * len(y_train))
    model.fit(x_train[:number_of_samples,:], y_train[:number_of_samples])

    predicted_train_y = model.predict(x_train)
    predicted_test_y = model.predict(x_test)

    train_error.append((predicted_train_y - train_y).std())
    test_error.append((predicted_test_y - test_y).std())


# # Random Forest estimator to the data

# In[66]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=5)
model.fit(x_train, y_train)
predicted_test_y = model.predict(x_test)
rms_random_forest = scatter_y(y_test, predicted_test_y)


# # Optimize model parameters

# In[67]:


model = RandomForestRegressor(n_estimators=100)


# In[69]:


n_features = x.shape[1]


# In[74]:


from sklearn.model_selection import RandomizedSearchCV
#from sklearn.grid_search import RandomizedSearchCV
grid = RandomizedSearchCV(model, n_iter=20, 
            param_distributions=dict(
                                          max_depth=np.arange(5,20+1), 
                                          max_features=np.arange(1, n_features+1)
                                    )
         )
grid.fit(x, y)
print(grid.best_params_)


# In[77]:


model = RandomForestRegressor(max_features=grid.best_params_["max_features"],
                              max_depth=grid.best_params_["max_depth"])
model.fit(x_train, y_train)
predicted_test_y = model.predict(x_test)
rms_optimized_random_forest = scatter_y(y_test, predicted_test_y)


# In[ ]:




