#!/usr/bin/env python
# coding: utf-8

# In[8]:


# dataset link = https://www.kaggle.com/uciml/breast-cancer-wisconsin-data


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data = pd.read_csv(r'C:\Users\SANDIP\Downloads\archive (5)\data.csv')

data.head()


# In[5]:


data.info()


# In[6]:


data.keys()


# In[7]:


data.isnull().sum()


# In[8]:


data.describe()


# In[9]:


data = data.drop(columns='Unnamed: 32')


# In[10]:


data.shape


# In[11]:


data.isnull().values.any()


# In[12]:


# Dealing with categorical values...


# In[13]:


data.select_dtypes(include='object').columns


# In[14]:


data['diagnosis'].unique()


# In[15]:


# One-Hot Encoding..


# In[16]:


data = pd.get_dummies(data=data, drop_first=True)


# In[17]:


data.head()


# In[18]:


sns.countplot(data['diagnosis_M'], label='Count')
plt.show()


# In[19]:


(data['diagnosis_M']==0).sum()


# In[20]:


(data['diagnosis_M']==1).sum()


# In[21]:


data2 = data.drop(columns='diagnosis_M')


# In[22]:


data2.head()


# In[23]:


data2.corr()


# In[50]:


pt = data2.corrwith(data['diagnosis_M'])
pt.plot.bar(figsize=(20,10), grid=True)


# In[26]:


# Above both graphs are showing that all the variables are independent...


# In[32]:


x = data.iloc[:, 1:-1].values


# In[33]:


x.shape


# In[38]:


y = data['diagnosis_M']

y.shape


# In[36]:


from sklearn.model_selection import train_test_split


# In[64]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)


# In[65]:


x_train.shape


# In[66]:


x_test.shape


# In[67]:


y_train.shape


# In[68]:


y_test.shape


# In[69]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


# In[70]:


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[81]:


from sklearn.linear_model import LogisticRegression


# In[82]:


lr = LogisticRegression()

lr.fit(x_train, y_train)


# In[83]:


y_pred = lr.predict(x_test)


# In[84]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[85]:


acc = accuracy_score(y_test, y_pred)
acc


# In[86]:


confusion_matrix(y_test, y_pred)


# In[87]:


from sklearn.model_selection import cross_val_score


# In[94]:


accuracy = cross_val_score(estimator=lr, X=x_train, y=y_train)


# In[95]:


accuracy.mean()


# In[96]:


accuracy.std()


# In[ ]:




