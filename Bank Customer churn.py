#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install "numpy>=1.18.5,<1.25.0"


# In[2]:


import pandas as pd


# In[3]:


import numpy as np


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


import seaborn as sns


# In[6]:


df = pd.read_csv(r"C:\Users\91860\Documents\bank customer churn\Bank Customer Churn Prediction.csv")


# In[7]:


df.head()


# In[8]:


df.info


# In[9]:


df.duplicated('customer_id').sum()


# In[10]:


df = df.set_index('customer_id')


# In[11]:


df.info()


# In[12]:


df['country'].value_counts()


# In[13]:


df.replace({'country': {'France': 2, 'Germany' : 1, 'Spain' : 0}}, inplace=True)


# In[14]:


df['gender'].value_counts()


# In[15]:


df.replace({'gender': {'Male' : 0, 'Female' : 1}}, inplace=True)


# In[16]:


df['products_number'].value_counts()


# In[17]:


df.replace({'product_number' : {1: 0, 2: 1, 3: 1, 4:1 }}, inplace=True)


# In[18]:


df['credit_card'].value_counts()


# In[19]:


df['active_member'].value_counts()


# In[20]:


df.loc[(df['balance']==0), 'churn'].value_counts()


# In[21]:


df['estimated_salary'] = np.where(df['balance']>0, 1, 0)


# In[22]:


df['estimated_salary'].hist()


# In[23]:


df.groupby(['churn' , 'country']).count()


# In[24]:


df.columns


# In[25]:


X = df.drop(['credit_score', 'churn'], axis = 1)


# In[26]:


y = df['churn']


# In[27]:


X.shape , y.shape


# In[28]:


df['churn'].value_counts()


# In[29]:


sns.countplot(x = 'churn', data = df);


# In[30]:


X.shape, y.shape


# In[31]:


get_ipython().system('pip install imblearn')


# In[32]:


from imblearn.under_sampling import RandomUnderSampler


# In[33]:


rus = RandomUnderSampler(random_state=2529)


# In[34]:


X_rus, y_rus = rus.fit_resample(X,y)


# In[35]:


X_rus.shape, y_rus.shape, X.shape, y.shape


# In[36]:


y.value_counts()


# In[37]:


y_rus.value_counts()


# In[38]:


y_rus.plot(kind = 'hist')


# In[39]:


from imblearn.over_sampling import RandomOverSampler


# In[40]:


ros = RandomOverSampler(random_state=2529)


# In[41]:


X_ros, y_ros = ros.fit_resample(X,y)


# In[42]:


X_ros.shape, y_ros.shape, X.shape, y.shape


# In[43]:


y.value_counts()


# In[44]:


y_ros.value_counts()


# In[45]:


y_ros.plot(kind = 'hist')


# In[46]:


get_ipython().system('pip install scikit-learn')


# In[47]:


from sklearn.model_selection import train_test_split


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=2529)


# In[49]:


X_train_ros, X_test_ros, y_train_ros, y_test_ros = train_test_split(X,y, test_size=0.3, random_state=2529)


# In[50]:


X_train_rus, X_test_rus, y_train_rus, y_test_rus = train_test_split(X,y, test_size=0.3, random_state=2529)


# In[51]:


from sklearn.preprocessing import StandardScaler


# In[52]:


sc= StandardScaler()


# In[53]:


X_train[[ 'age', 'tenure', 'balance', 'estimated_salary']] = sc.fit_transform(X_train[[ 'age', 'tenure', 'balance', 'estimated_salary']])


# In[54]:


X_test[[ 'age', 'tenure', 'balance', 'estimated_salary']] = sc.fit_transform(X_test[[ 'age', 'tenure', 'balance', 'estimated_salary']])


# In[55]:


X_train_rus[[ 'age', 'tenure', 'balance', 'estimated_salary']] = sc.fit_transform(X_train_rus[[ 'age', 'tenure', 'balance', 'estimated_salary']])


# In[56]:


X_test_rus[[ 'age', 'tenure', 'balance', 'estimated_salary']] = sc.fit_transform(X_test_rus[[ 'age', 'tenure', 'balance', 'estimated_salary']])


# In[57]:


X_train_ros[[ 'age', 'tenure', 'balance', 'estimated_salary']] = sc.fit_transform(X_train_ros[[ 'age', 'tenure', 'balance', 'estimated_salary']])


# In[58]:


X_test_ros[[ 'age', 'tenure', 'balance', 'estimated_salary']] = sc.fit_transform(X_test_ros[[ 'age', 'tenure', 'balance', 'estimated_salary']])


# In[59]:


from sklearn.svm import SVC


# In[60]:


svc = SVC()


# In[61]:


svc.fit(X_train, y_train)


# In[62]:


y_pred = svc.predict(X_test)


# In[63]:


get_ipython().system('pip install scikit-learn')


# In[64]:


from sklearn.metrics import confusion_matrix, classification_report


# In[65]:


confusion_matrix(y_test, y_pred)


# In[66]:


print(classification_report(y_test, y_pred))


# In[67]:


from sklearn.model_selection import GridSearchCV


# In[68]:


param_grid = {'C': [0.1,1,10],
             'gamma': [1,0.1,0.01],
             'kernel': ['rbf'],
             'class_weight': ['balanced']}


# In[69]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2, cv =2)
grid.fit(X_train, y_train)


# In[70]:


print(grid.best_estimator_)


# In[71]:


grid_predictions = grid.predict(X_test)


# In[72]:


confusion_matrix(y_test,grid_predictions)


# In[73]:


print(classification_report(y_test,grid_predictions))


# In[74]:


svc_rus = SVC()


# In[75]:


svc_rus.fit(X_train_rus, y_train_rus)


# In[76]:


y_pred_rus = svc_rus.predict(X_test_rus)


# In[77]:


confusion_matrix(y_test_rus, y_pred_rus)


# In[78]:


print(classification_report(y_test_rus, y_pred_rus))


# In[79]:


param_grid = {'C': [0.1,1,10],
             'gamma': [1,0.1,0.01],
             'kernel': ['rbf'],
             'class_weight': ['balanced']}


# In[80]:


grid_rus = GridSearchCV(SVC(),param_grid,refit=True,verbose=2, cv =2)
grid_rus.fit(X_train_rus, y_train_rus)


# In[81]:


print(grid_rus.best_estimator_)


# In[82]:


grid_predictions_rus = grid_rus.predict(X_test_rus)


# In[83]:


confusion_matrix(y_test_rus, grid_predictions_rus)


# In[84]:


print(classification_report(y_test_rus,grid_predictions_rus))


# In[85]:


svc_ros = SVC()


# In[86]:


svc_ros.fit(X_train_ros, y_train_ros)


# In[87]:


y_pred_ros = svc_ros.predict(X_test_ros)


# In[88]:


confusion_matrix(y_test_ros, y_pred_ros)


# In[89]:


print(classification_report(y_test_ros, y_pred_ros))


# In[90]:


param_grid = {'C': [0.1,1,10],
             'gamma': [1,0.1,0.01],
             'kernel': ['rbf'],
             'class_weight': ['balanced']}


# In[91]:


grid_ros = GridSearchCV(SVC(),param_grid,refit=True,verbose=2, cv =2)
grid_ros.fit(X_train_ros, y_train_ros)


# In[92]:


print(grid_ros.best_estimator_)


# In[93]:


grid_predictions_ros = grid_ros.predict(X_test_ros)


# In[94]:


confusion_matrix(y_test_ros, grid_predictions_ros)


# In[95]:


print(classification_report(y_test_ros, grid_predictions_ros))


# In[96]:


print(classification_report(y_test, y_pred))


# In[97]:


print(classification_report(y_test, grid_predictions))


# In[98]:


print(classification_report(y_test_rus, y_pred_rus))


# In[99]:


print(classification_report(y_test_ros, y_pred_ros))


# In[100]:


print(classification_report(y_test_ros, grid_predictions_ros))


# In[ ]:




