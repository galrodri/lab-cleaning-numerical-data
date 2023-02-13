#!/usr/bin/env python
# coding: utf-8

# # LAB | Cleaning Numerical Data

# ### Step 1 - Import libraries

# In[1]:


# Importing common libraries
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import statistics
import datetime

# Importing libraries for data visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.colors import BASE_COLORS

# Importing libraries to ignore warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


# ### Step 2 - Load .csv file into the variable customer_df and analysing main features

# In[2]:


customer_df = pd.read_csv(r'C:\Users\galrodri\Documents\GitHub\lab-cleaning-numerical-data\files_for_lab\we_fn_use_c_marketing_customer_value_analysis.csv')
customer_df.head()


# In[3]:


customer_df.shape


# In[4]:


customer_df.info()


# ### Step 3 - Formatting columns

# In[5]:


customer_df.columns = map(lambda x: x.lower().replace("-", "_").replace(" ", "_"), customer_df.columns) # lowering headers


# In[6]:


print(customer_df.columns)


# ### Step 5 - Data types

# In[7]:


# checking data types for our dataset
customer_df.dtypes


# In[8]:


# Modifying data type for Effective to Date
customer_df['effective_to_date'] = pd.to_datetime(customer_df['effective_to_date'], errors='coerce')


# In[9]:


# Checking NaN values (step 10) and there are none
customer_df.isna().sum()


# ### Step 6 - Correlation matrix

# In[10]:


mask=np.zeros_like(customer_df.corr())
mask[np.triu_indices_from(mask)]=True
fig, ax=plt.subplots(figsize=(14, 8))
ax=sns.heatmap(customer_df.corr(), mask=mask, annot=True)
plt.show()


# We identify a negative correlation between the total claim amount and the income.
# We also identify positive correlation of total claim amount with the monthly premium auto, meaning that customers who pay a higher premium tend to claim a larger amount

# ### Step 7 - Visualizing continous variables

# In[11]:


numericals = customer_df.select_dtypes(np.number)
numericals.head()


# In[12]:


for col in numericals.columns:
    sns.displot(data=numericals, x=col)
    plt.show()


# ### Step 8 - Visualizing categorical variables

# In[13]:


categoricals = customer_df.select_dtypes(np.object)
categoricals.head()


# In[22]:


plt.figure(figsize=(10,6))
sns.countplot('sales_channel', hue='response', data=categoricals)
plt.ylabel('Response by Sales Channel')
plt.show()


# In[24]:


plt.figure(figsize=(10,6))
sns.countplot('state', data=categoricals)
plt.ylabel('Total number of customers per State')
plt.show()


# In[27]:


plt.figure(figsize=(10,6))
sns.countplot('employmentstatus', hue='gender', data=categoricals)
plt.ylabel('Employment Status per Gender')
plt.show()


# In[30]:


plt.figure(figsize=(10,6))
sns.countplot('vehicle_class', hue='coverage', data=categoricals)
plt.ylabel('Type of Coverage based on Vehicle Class')
plt.show()


# In[33]:


plt.figure(figsize=(10,6))
sns.countplot('location_code', hue='policy_type', data=categoricals)
plt.ylabel('Policy type based on location')
plt.show()


# ### Step 9 - Identify outliers in continous variables

# In[35]:


# We split numerical variables into continous and discrete
discrete = [i for i in numericals if (len(numericals[i].unique()) < (numericals.shape[0] * 0.01))]
discrete


# In[36]:


continuous = list(numericals.drop(columns = discrete).columns)
continuous


# In[38]:


continuous_df = numericals.drop(columns = discrete)
continuous_df.head()


# In[44]:


f, ax = plt.subplots(1, 5, figsize=(16,8))


for i, col in enumerate(continuous[:5]):
    sns.boxplot(data = continuous_df[col], ax = ax[i])
    ax[i].set_title(col, fontsize = 14)
plt.show();

