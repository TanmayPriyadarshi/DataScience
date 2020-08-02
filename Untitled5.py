#!/usr/bin/env python
# coding: utf-8

# # How to project 2D Non-linearly Separable data into 3D so that it can be linearly separable...

# # For that purpose all you need to do is to project the data into the 1-more extra dimension, which is nothing but the combination of the previous dimensions. More frequently in this example we have 2D non-linear data which can be linearly separable if we project that data into 3D.

# In[1]:


from sklearn.datasets import make_circles
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# In[16]:


x,y = make_circles(n_samples=500,noise=0.01)


# In[3]:


print(x.shape,y.shape)


# In[17]:


plt.scatter(x[:,0],x[:,1],c=y)
plt.show()


# In[18]:


def phi(x):
    x1 = x[:,0]
    x2 = x[:,1]
    x3 = x1**2 + x2**2
    
    x_ = np.zeros((x.shape[0],3))
    print(x_.shape)
    x_[:,:-1]=x
    x_[:,-1]=x3
    return x_


# In[19]:


x_ = phi(x)
print(x_)


# In[20]:


def plotting3d(x):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,projection='3d')
    x1 = x[:,0]
    x2 = x[:,1]
    x3 = x[:,2]
    ax.scatter(x1,x2,x3,zdir='z',s=20,c=y,depthshade=True)
    plt.show()
    return ax


# In[21]:


plotting3d(x_)


# In[ ]:





# In[ ]:





# In[12]:


fig = plt.figure()
#fig.add_subplot(1, 2, 1)   #top and bottom left
#fig.add_subplot(2, 2, 2)   #top right
fig.add_subplot(2, 2, 4)   #bottom right 
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




