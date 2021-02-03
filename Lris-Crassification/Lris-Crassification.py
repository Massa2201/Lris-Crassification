#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -U scikit-learn')


# In[1]:


import sklearn


# In[ ]:


print(sklearn.__version__)


# In[2]:


import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()


# In[ ]:


# アヤメの特徴量の名称
print(iris.feature_names)


# In[3]:


import pandas as pd

pd.DataFrame(iris.data, columns=iris.feature_names)


# In[ ]:


print(iris.data)


# In[ ]:


# 最初の:は行列の行を指定する，2つ目の:は列を指定している
first_one_feature = iris.data[:, :1]
pd.DataFrame(first_one_feature, columns=iris.feature_names[:1])


# In[ ]:


# 2つの列を抽出して比較する
first_two_features = iris.data[:, :2]
print(first_two_features)


# In[ ]:


last_two_features = iris.data[:, 2:]
print(last_two_features)


# In[7]:


teacher_labels = iris.target
print(teacher_labels)


# In[8]:


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


# In[9]:


x_min, x_max = all_features[:, 2].min(), all_features[:, 3].max()
y_min, y_max = all_features[:, 2].min(), all_features[:, 3].max()

plt.figure(2, figsize= (12, 9))
plt.clf()

plt.scatter(all_features[:, 2], all_features[:, 3], s = 300, c = teacher_labels, cmap = plt.cm.Set2, edgecolor = "darkgray")
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.show()


# In[10]:


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


# In[2]:


from sklearn.svm import SVC

import numpy as np
import sklearn.datasets as datasets
import matplotlib.pyplot as plt


# In[3]:


iris = datasets.load_iris()

# 2つの特徴を2次元データへ
first_two_features = iris.data[:, [0, 1]]
teacher_labels = iris.target

first_two_features = first_two_features[teacher_labels != 2]
teacher_labels = teacher_labels[teacher_labels != 2]


# In[4]:


model = SVC(C = 1.0, kernel = 'linear')

model.fit(first_two_features, teacher_labels)


# In[5]:


# 作図を行うため回帰係数を確認する
print(model.coef_)
# 誤差
print(model.intercept_)


# In[7]:


# figureオブジェクト作成サイズの決定
fig, ax = plt.subplots(figsize = (12, 9))

# ---------------------------------------------------------------------
# 花のデータを描画する
setosa = first_two_features[teacher_labels == 0]
versicolor = first_two_features[teacher_labels == 1]
plt.scatter(setosa[:, 0], setosa[:, 1], s=300, c='blue', linewidths=0.5, edgecolors='lightgray')
plt.scatter(versicolor[:, 0], versicolor[:, 1], s=300, c='firebrick', linewidths=0.5, edgecolors='lightgray')
# ---------------------------------------------------------------------

Xi = np.linspace(4, 7.25)
Y = -model.coef_[0][0] / model.coef_[0][1] * Xi - model.intercept_ / model.coef_[0][1]

ax.plot(Xi, Y, linestyle = 'dashed', linewidth = 3)

plt.show()


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
iris = datasets.load_iris()

last_two_features = iris.data[:, [2, 3]]
teacher_labels = iris.target

train_features, test_features, train_teacher_labels, test_teacher_labels = train_test_split(last_two_features,
                                                                                            
                                                                                            teacher_labels,
                                                                                            
                                                                                            test_size = 0.2,
                                                                                            
                                                                                            random_state = None)

sc = StandardScaler()
sc.fit(train_features)

train_features_std = sc.transform(train_features)

test_features_std = sc.transform(test_features)

from sklearn.svm import SVC

# svmのインスタンスを作成する
model = SVC(kernel = 'linear', random_state = None)

# モデルの学習
model.fit(train_features_std, train_teacher_labels)


# In[9]:


from sklearn.metrics import accuracy_score

predict_train = model.predict(train_features_std)

accuracy_train = accuracy_score(train_teacher_labels, predict_train)
print('学習データに対する分類精度 : %.2f' % accuracy_train)


# In[ ]:




