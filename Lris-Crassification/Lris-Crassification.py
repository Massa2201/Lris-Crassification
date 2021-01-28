#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -U scikit-learn')


# In[4]:


import sklearn


# In[4]:


print(sklearn.__version__)


# In[5]:


import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()


# In[8]:


# アヤメの特徴量の名称
print(iris.feature_names)


# In[6]:


import pandas as pd

pd.DataFrame(iris.data, columns=iris.feature_names)


# In[11]:


print(iris.data)


# In[12]:


# 最初の:は行列の行を指定する，2つ目の:は列を指定している
first_one_feature = iris.data[:, :1]
pd.DataFrame(first_one_feature, columns=iris.feature_names[:1])


# In[13]:


# 2つの列を抽出して比較する
first_two_features = iris.data[:, :2]
print(first_two_features)


# In[14]:


last_two_features = iris.data[:, 2:]
print(last_two_features)


# In[12]:


teacher_labels = iris.target
print(teacher_labels)


# In[9]:


# 1, 2列目
all_features = iris.data
x_min, x_max = all_features[:, 0].min(), all_features[:, 0].max()
y_min, y_max = all_features[:, 1].min(), all_features[:, 1].max()

plt.figure(2, figsize = (12, 9))
plt.clf()

plt.scatter(all_features[:, 0], all_features[:, 1], s=300, c=teacher_labels, cmap = plt.cm.Set2, edgecolor = "darkgray")

plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.grid(True)


# In[14]:


x_min, x_max = all_features[:, 2].min(), all_features[:, 3].max()
y_min, y_max = all_features[:, 2].min(), all_features[:, 3].max()

plt.figure(2, figsize= (12, 9))
plt.clf()

plt.scatter(all_features[:, 2], all_features[:, 3], s = 300, c = teacher_labels, cmap = plt.cm.Set2, edgecolor = "darkgray")
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.show()


# In[21]:


#  すべての特徴量を使う，PCA処理により4つの特徴量を3つにする
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

# 12, 9 = 横，縦
fig = plt.figure(1, figsize = (12, 9))
# -140, 100 = そろぞれ上下の回転率と考えられる
ax = Axes3D(fig, elev = -140, azim = 100)

reduced_features = PCA(n_components = 3).fit_transform(all_features)
ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], c = teacher_labels, cmap = plt.cm.Set2, edgecolor = "darkgray", s = 200)
plt.show()


# In[ ]:




