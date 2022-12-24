#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


housing = pd.read_csv('C:\Users\fatma_000\Desktop\archive\train')


# In[ ]:


housing.head()


# In[ ]:


housing.info()


# In[ ]:


housing.hist(bins=50,figsize=(20,15))
plt.show()


# In[ ]:


housing.isnull().sum()


# In[ ]:


plt.subplots(figsize=(15, 7))

plt.title('Histogram Plot: Total Bedrooms')

total_bedrooms = housing['total_bedrooms']

plt.hist( total_bedrooms, bins=500, alpha=0.8,
          histtype='bar', color='steelblue',
          edgecolor='green')

plt.show()


# In[ ]:


plt.subplots(figsize=(15, 7))

plt.title('Histogram Plot: Households')

households = housing['households']

plt.hist( households, bins=500, alpha=0.8,
          histtype='bar', color='blue',
          edgecolor='green')

plt.show()


# In[ ]:


plt.figure(figsize=(14, 6))

plt.rcParams['axes.grid'] = False # For suppressing the depreciation error

plt.title('Scatter Plot: Total Bedrooms and Households ')


N = households.size
colors = np.random.rand(N)
area = np.pi * (20 * np.random.rand(N))**2 

plt.xlabel('Total Bedrooms')
plt.ylabel('Households')

plt.scatter(total_bedrooms, households, s=area, c=colors, alpha=0.5, cmap='Spectral')
plt.colorbar()

plt.show()


# In[ ]:




