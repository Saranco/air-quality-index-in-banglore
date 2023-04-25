#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('train_aqi.csv')


# In[3]:


df


# In[6]:


df.isnull().sum()


# In[7]:


df.info()


# In[8]:


quali=[x for x in df.columns if df[x].dtype=='O']


# In[9]:


quali


# In[11]:


conti=[x for x in df.columns if df[x].dtype!='O'and len(df[x].unique())>25]


# In[12]:


conti


# In[13]:


disc=[x for x in df.columns if df[x].dtype!='O' and len(df[x].unique())<25]


# In[14]:


disc


# In[17]:


for x in conti:
    sns.histplot(df[x],kde=True,color="Red")
    plt.show()


# In[18]:


left_skew=[x for x in conti if df[x].skew()<0]


# In[19]:


right_skew=[x for x in conti if df[x].skew()>0]


# In[21]:


print(left_skew,right_skew)


# In[24]:


plato=[x for x in conti if df[x].kurtosis()<3]


# In[25]:


meso=[x for x in conti if df[x].kurt()==0]


# In[26]:


lepto=[x for x in conti if df[x].kurt()>3 ]


# In[27]:


print(plato,meso,lepto)


# In[28]:


for x in conti:
    df.boxplot(column=x)
    plt.show()


# In[30]:


df[conti].describe()


# In[40]:


for x in quali[1:]:
    df.groupby(x)['air_pollution_index'].median().sort_values().plot.barh()
    plt.show()


# In[45]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,cmap='RdYlGn')
plt.show


# In[47]:


from sklearn.preprocessing import LabelEncoder


# In[48]:


enco_is_holi=LabelEncoder()


# In[49]:


enco_is_holi.fit(df['is_holiday'])


# In[50]:


df['is_holiday']=enco_is_holi.transform(df['is_holiday'])


# In[53]:


df[quali]


# In[54]:


enco_wea=LabelEncoder()


# In[55]:


enco_wea.fit(df['weather_type'])


# In[56]:


df['weather_type']=enco_wea.transform(df['weather_type'])


# In[57]:


df[quali]


# In[61]:


x=df.iloc[:,1:-2]


# In[59]:


y=df.iloc[:,-2]


# In[62]:


x


# In[63]:


y


# In[64]:


from sklearn.model_selection import train_test_split


# In[68]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[72]:


from sklearn.preprocessing import StandardScaler


# In[73]:


ss=StandardScaler()


# In[78]:


x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)


# In[69]:


from sklearn.linear_model import LinearRegression


# In[70]:


ln=LinearRegression()


# In[86]:


ln_model=ln.fit(x_train,y_train)


# In[84]:


pr=ln_model.predict(x_test)


# In[82]:


pr


# In[3]:


81%3


# In[4]:


78%3


# In[ ]:




