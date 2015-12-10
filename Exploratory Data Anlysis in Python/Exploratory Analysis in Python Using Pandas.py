
# coding: utf-8

# In[1]:

# Importing libraries and the data set


# In[7]:

import pandas as pd
import numpy as np
import matplotlib as plt


# In[8]:

# Reading the dataset in a dataframe using Pandas
df = pd.read_csv("E:/Data Collection/Titanic Dataset_Kaggle/train.csv")


# In[9]:

# Some quick data exploration
df.head(10)  #Printing the dataset to explore first 10 row datavisually,


# In[10]:

# look at summary of numerical fields by using describe() function
# It Printout summary statistics for numerical fields
df.describe()


# In[11]:

#also look at the median of these variables and compare them with mean
#to see possible skew in the dataset
df['Age'].median()


# In[12]:

#For the non-numerical values (e.g. Sex, Embarked etc.)
#we can look at unique values to understand whether they 
#make sense or not
df['Sex'].unique()


# In[8]:

#Distribution analysis
#familiar with basic data characteristics 
#distribution of various variables 


# In[13]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[10]:

#plot histograms for "Age" and "Fare" using the following commands


# In[21]:

#Plot histogram of "Age"
fig=plt.figure()
ax = fig.add_subplot(111)
ax.hist(df['Age'],bins=10,range= (df['Age'].min(),df['Age'].max()))
plt.title('Age of Distribution')
plt.xlabel('Age')
plt.ylabel('Count of Passengers')
plt.show()


# In[22]:

#Plot histogram of "Fare"
fig=plt.figure()
ax = fig.add_subplot(111)
ax.hist(df['Fare'],bins=10,range= (df['Fare'].min(),df['Fare'].max()))
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Count of Passengers')
plt.show()


# In[ ]:

#Box plots to understand the distributions.
#Box plot for "Fare" can be plotted by:


# In[24]:

fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot(df['Fare'])
plt.show()


# In[28]:

#looking at fare across the 3 passenger classes. 
#Let us segregate them by Passenger class

df.boxplot(column='Fare', by ='Pclass')


# #####Clearly, both Age and Fare require some amount of data munging. Age has about 31% missing values, while Fare has a few Outliers, which demand deeper understanding

# ### Categorical variable analysis:

# ######Following code plots the distribution of population by PClass and their probability of survival

# In[44]:

temp1 = df.groupby('Pclass').Survived.count()
temp2 = df.groupby('Pclass').Survived.sum()/df.groupby('Pclass').Survived.count()
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Pclass')
ax1.set_ylabel('Count of Passengers')
ax1.set_title("Passengers by Pclass")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Pclass')
ax2.set_ylabel('Probability of Survival')
ax2.set_title("Probability of survival by class")



# #### These two plots can also be visualized by combining them in a stacked chart:

# In[45]:

temp3 = pd.crosstab([df.Pclass, df.Sex], df.Survived.astype(bool))
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

