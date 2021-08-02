#!/usr/bin/env python
# coding: utf-8

# # AD CLICK PREDICTION
# 

# ![ddd.png](attachment:ddd.png)
The dataset is composed of:

-Daily Time Spent on Site: Amount of time in the website
-Age: Customer age
-Area Income: Average revenue of customer 
-Daily Internet Usage: daily average time on internet
-Ad Topic Line: Text of the advertissement
-City: City of the customer
-Male: Wheter or not user is a male
-Country: country of the user
-Timestamp: Time at which consumer clicked on Ad or closed window
-Clicked on Ad: 0 or 1 indicated clicking on Ad
# # importing libraries

# In[150]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")


# In[151]:


#Change working directory
import os
os.chdir("C:\\Users\\shikh\\Downloads")


# In[152]:


#Loading dataset
df = pd.read_csv("advertising.csv")


# In[153]:


#print first 5 
print('First 5 observations:')
print('')
print(df.head())


# In[154]:


#shape of data
print('shape:',df.shape)


# In[155]:


df.info()


# In[156]:


df.describe()


# # EDA

# In[157]:


#checking unique values for each column

for col in df:
    print(f'number of unique values of {col}:',df[col].nunique())


# In[158]:


#checking for missing values
df.isnull().sum()


# In[169]:


print(df['Clicked on Ad'].value_counts())
sns.countplot(x='Clicked on Ad',data=df)


# In[160]:


#deleting columns city, country
df_final = df.drop(['Timestamp', 'Ad Topic Line', 'Country', 'City'], axis=1)


# In[161]:


df_final.head()


# In[162]:





# In[163]:


sns.pairplot(df_final,hue='Clicked on Ad')


# In[164]:


df_final.corr()


# In[170]:


plt.figure(figsize=(12, 10))
sns.heatmap(df_final.corr())


# # Logistic Regression
# 

# Split the data into training set and testing set using train_test_split

# In[166]:


from sklearn.model_selection import train_test_split


# In[172]:


X = df_final[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
Y = df_final['Clicked on Ad']


# In[173]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[174]:


print('Number of data points in train data:', X_train.shape)
print('Number of data points in test data:', Y_test.shape)


# Train a logistic regression model on the training set.

# In[175]:


from sklearn.linear_model import LogisticRegression


# In[181]:


import warnings
LR = LogisticRegression(solver='liblinear')
LR.fit(X_train,Y_train)


# In[205]:


from sklearn import metrics
predictions = LR.predict(X_test)
predictions_train = LR.predict(X_train)
print ('Accuracy Score:',metrics.accuracy_score(Y_test, predictions))
print('')
print('Log Loss:',metrics.log_loss(Y_test, predictions))
print('')
print('Confusion Matrix:\n',metrics.confusion_matrix(Y_test, predictions))
print('')
print('Classification report:\n\n',metrics.classification_report(Y_test, predictions))


# In[183]:


Probabilities = LR.predict_proba(X_test)


# The end
