#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("CarPrice_Assignment.csv")


# In[3]:


df


# In[4]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[5]:


df.head()


# In[6]:


df.isnull().sum()


# In[7]:


from sklearn import linear_model


# In[8]:


df=df.drop(['car_ID'],axis='columns')


# In[9]:


target=df.price
target


# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[11]:


from word2number import w2n


# In[12]:


df.doornumber = df.doornumber.apply(w2n.word_to_num)
df


# In[13]:


df.cylindernumber = df.cylindernumber.apply(w2n.word_to_num)
df


# In[14]:


inputs=df.drop(['price','CarName'],axis='columns')
inputs


# In[15]:


le_fueltype=LabelEncoder()
le_aspiration=LabelEncoder()
le_carbody=LabelEncoder()
le_drivewheel=LabelEncoder()
le_enginelocation=LabelEncoder()
le_enginetype=LabelEncoder()
le_fuelsystem=LabelEncoder()


# In[16]:


inputs['fueltype_n']=le_fueltype.fit_transform(inputs['fueltype'])
inputs=inputs.drop('fueltype',axis='columns')
inputs


# In[17]:


inputs['aspiration_n']=le_aspiration.fit_transform(inputs['aspiration'])
inputs=inputs.drop('aspiration',axis='columns')


# In[18]:


inputs['carbody_n']=le_fueltype.fit_transform(inputs['carbody'])
inputs=inputs.drop('carbody',axis='columns')


# In[19]:


inputs['drivewheel_n']=le_fueltype.fit_transform(inputs['drivewheel'])
inputs=inputs.drop('drivewheel',axis='columns')


# In[20]:


inputs['enginelocation_n']=le_fueltype.fit_transform(inputs['enginelocation'])
inputs=inputs.drop('enginelocation',axis='columns')


# In[21]:


inputs['enginetype_n']=le_fueltype.fit_transform(inputs['enginetype'])
inputs=inputs.drop('enginetype',axis='columns')


# In[22]:


inputs['fuelsystem_n']=le_fueltype.fit_transform(inputs['fuelsystem'])
inputs=inputs.drop('fuelsystem',axis='columns')


# In[23]:


inputs=inputs.drop(["fueltype_n"],axis='columns')
inputs


# In[27]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train,X_test,Y_train,Y_test=train_test_split(inputs,target,test_size=0.2)


# In[31]:


len(X_train)


# In[32]:


len(X_test)


# In[34]:


model=linear_model.LinearRegression()


# In[37]:


model.fit(X_train,Y_train)


# In[38]:


model.score(X_train,Y_train)


# In[39]:


model.score(X_train,Y_train)


# In[ ]:




