#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


df = pd.read_csv('heart1.csv')
df.head()


# In[17]:


df.shape


# In[18]:


df.keys()


# In[19]:


df.info()


# In[20]:


df.describe()


# In[21]:


df


# In[22]:


df.isna().sum()


# In[23]:



df.dropna(axis = 0, inplace = True) 
print(df.shape)


# In[24]:


df['TenYearCHD'].value_counts()


# In[25]:


plt.figure(figsize = (14, 10)) 
sns.heatmap(df.corr(), cmap='Purples',annot=True, linecolor='Green', linewidths=1.0)
plt.show()


# In[26]:


sns.pairplot(df)
plt.show()


# In[27]:


sns.catplot(data=df, kind='count', x='male',hue='currentSmoker')
plt.show()


# In[28]:


sns.catplot(data=df, kind='count', x='TenYearCHD', col='male',row='currentSmoker', palette='Blues')
plt.show()


# In[29]:


X = df.iloc[:,0:15]
y = df.iloc[:,15:16]


# In[30]:



X.head()


# In[31]:



y.head()


# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=21)


# In[33]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[34]:


logreg.fit(X_train, y_train)


# In[35]:


y_pred = logreg.predict(X_test)


# In[36]:


score = logreg.score(X_test, y_test)
print("Prediction score is:",score)


# In[37]:



from sklearn.metrics import confusion_matrix, classification_report 
cm = confusion_matrix(y_test, y_pred) 
print("Confusion Matrix is:\n",cm)


# In[38]:


print("Classification Report is:\n\n",classification_report(y_test,y_pred))


# In[39]:


conf_matrix = pd.DataFrame(data = cm,  
                           columns = ['Predicted:0', 'Predicted:1'],  
                           index =['Actual:0', 'Actual:1']) 
plt.figure(figsize = (10, 6)) 
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "Greens", linecolor="Blue", linewidths=1.5) 
plt.show()


# In[ ]:




