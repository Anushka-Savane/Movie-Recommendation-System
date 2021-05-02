#!/usr/bin/env python
# coding: utf-8

# ## <span style='background:yellow'> Movie Recommendation </span>

# In[2]:


import numpy as np
import pandas as pd


# <span style='background:yellow'> The two dataset and its merge </span>

# In[3]:


x = pd.read_csv(r'C:\Users\Admin\Desktop\ANUSHKA\MovieRecommendation\ml-latest-small\ml-latest-small\ratings.csv')

y = pd.read_csv(r'C:\Users\Admin\Desktop\ANUSHKA\MovieRecommendation\ml-latest-small\ml-latest-small\movies.csv')

df = pd.merge(x, y, on='movieId')
df.head()


# In[4]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('white')


# In[5]:


df.groupby('title')['rating'].mean().sort_values(ascending=False) #ratings ka mean from all users with title in descending order(it isnt written descending=True cause sort_value dosent have a keyword named descending)


df.groupby('title')['rating'].count().sort_values(ascending=False) #title with the count of ratings 

ratings = pd.DataFrame(df.groupby('title')['rating'].mean()) #title and ratings in order of dataset

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings


# <span style='background :yellow'> Countplot </span>

# In[27]:


plt.figure(figsize=(8,10)) #8=width,10=length
x = sns.countplot( y='rating', data=df)


# <span style='background :yellow' > Histogram </span>

# In[21]:


plt.figure(figsize = (10,4))
ratings['num of ratings'].hist(bins=70)


# In[7]:


plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)  #graph of ratings to num of users


# <span style='background :yellow' > Jointplot </span>

# In[8]:


sns.jointplot(x='rating', y='num of ratings', data=ratings,alpha=0.5) #aplha=0.5 is rgba value which is from 0 to 1, transperancy increases from 0 to 1


# In[28]:


df1 = df.pivot_table(index='userId',columns='title', values='rating')  #which userid has given rating to which movie using pivottable


# In[22]:


ToyStory_user_ratings = df1['Toy Story (1995)'] #toystory's ratings given by all userid's


# In[11]:


similar_to_ToyStory = df1.corrwith(ToyStory_user_ratings) #pairwise correlate between rows or columns


# In[23]:


corr_ToyStory = pd.DataFrame(similar_to_ToyStory, columns=['Correlation'])
 #correlating with all movies


# In[24]:


corr_ToyStory.dropna(inplace=True) #max correlation value is 1 higher the value it gets recommended the -1 correlation is said to be the strongest


# <span style='background:yellow'> Correlation with movies </span>

# In[14]:


corr_ToyStory.sort_values('Correlation', ascending = False)


# In[25]:


corr_ToyStory = corr_ToyStory.join(ratings['num of ratings'])


# <span style='background :yellow' > Recommended Movies </span>

# In[26]:


corr_ToyStory.head(10)[corr_ToyStory.head(10)['num of ratings']>5].sort_values('Correlation',ascending=False) #if num of ratings is >1 it will display and the correlations values will be in descng order


# In[17]:


Babe_user_ratings = df1['Babe (1995)']
similar_to_Babe = df1.corrwith(Babe_user_ratings)
corr_Babe = pd.DataFrame(similar_to_Babe, columns=['Correlation'])
corr_Babe.dropna(inplace=True)
corr_Babe = corr_Babe.join(ratings['num of ratings'])
corr_Babe.head(5)[corr_Babe.head(5)['num of ratings']>1].sort_values('Correlation',ascending=False)


# In[ ]:




