#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[100]:


df=pd.read_excel(r"E:\Brave Downloads\customer_churn_large_dataset.xlsx")


# In[101]:


df.head()


# In[102]:


df.duplicated().sum()


# In[103]:


df=df.iloc[:,df.columns!='Name']
df.head()



# In[104]:


# In[5]:


df.info()




# In[105]:


# In[6]:


df.describe()




# In[106]:


import matplotlib.pyplot as plt
import seaborn as sns

df=pd.get_dummies(df,drop_first=True)
df





# In[107]:


x=df.iloc[:,df.columns!='CustomerID']
x=x.iloc[:,x.columns!='Churn']


# In[108]:


x


# In[109]:


y=df['Churn']


# In[110]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[111]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[112]:


x_train=scaler.fit_transform(x_train)
x_train


# In[113]:


x_test=scaler.transform(x_test)


# In[114]:


x_test


# In[115]:


from sklearn.tree import DecisionTreeClassifier


# In[116]:


tree=DecisionTreeClassifier(max_depth=5,min_samples_split=5000)


# In[198]:


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=750,max_depth=30,min_samples_split=5000,max_features=6,criterion='entropy', random_state=42)


# In[199]:


forest.fit(x_train,y_train)


# In[200]:


pred=forest.predict(x_test)


# In[201]:


from sklearn.metrics import accuracy_score


# In[202]:


accuracy_score(y_test,pred)


# In[203]:


import pickle


# In[204]:


pickle.dump(forest,open('churn.pkl','wb'))


# In[ ]:




