#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

days = np.arange(1, 11)
weight = np.array([10, 14, 18, 20, 18, 16, 17, 18, 20, 17])

# 그래프를 설정합니다
plt.ylim([0, weight.max()+1])
plt.xlabel("days")
plt.ylabel("weight")

# 검은색의 원형 마커를 적용한 꺾은선 그래프를 작성하세요
plt.plot(days, weight, marker="o", markerfacecolor="k") 
plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

days = np.arange(1, 11)
weight = np.array([10, 14, 18, 20, 18, 16, 17, 18, 20, 17])

# 그래프를 설정합니다
plt.ylim([0, weight.max()+1])
plt.xlabel("days")
plt.ylabel("weight")

# 검은색의 원형 마커를 설정하고, 파란 점선을 적용한 꺾은선 그래프를 작성하세요
plt.plot(days, weight, linestyle="--", color="b", marker="o", markerfacecolor="k") 
plt.show()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

x = [1, 2, 3, 4, 5, 6]
y = [12, 41, 32, 36, 21, 17]

# 막대 그래프를 작성하세요
plt.bar(x, y)
plt.show()


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

x = [1, 2, 3, 4, 5, 6]
y = [12, 41, 32, 36, 21, 17]
labels = ["Apple", "Orange", "Banana", "Pineapple", "Kiwifruit", "Strawberry"]

# 막대 그래프를 만들고, 가로축에 라벨을 설정하세요
plt.bar(x, y, tick_label = labels)
plt.show()


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

x = [1, 2, 3, 4, 5, 6]
y1 = [12, 41, 32, 36, 21, 17]
y2 = [43, 1, 6, 17, 17, 9]
labels = ["Apple", "Orange", "Banana", "Pineapple", "Kiwifruit", "Strawberry"]

# 누적 막대 그래프를 작성하여 가로축에 라벨을 설정하세요
plt.bar(x, y1, tick_label=labels)
plt.bar(x, y2, bottom=y1)

# 계열에 라벨을 지정할 수 있습니다
plt.legend(("y1", "y2"))
plt.show()


# In[8]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(0)
data = np.random.randn(10000)

# data를 이용해 히스토그램을 작성하세요
plt.hist(data)
plt.show()


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(0)
data = np.random.randn(10000)

# 구간 수 100을 적용한 히스토그램을 작성하세요
plt.hist(data, bins = 100)
plt.show()


# In[12]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(0)
data = np.random.randn(10000)

# 정규화된, 구간 수 100인 히스토그램을 작성하세요
plt.hist(data, bins=100, normed=True)
plt.show()


# In[13]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(0)
data = np.random.randn(10000)

# 정규화된, 구간 수 100인 누적 히스토그램을 작성하세요
plt.hist(data, bins=100, normed=True, cumulative=True) 
plt.show()


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(0)
x = np.random.choice(np.arange(100), 100)
y = np.random.choice(np.arange(100), 100)

# 산포도를 작성하세요
plt.scatter(x, y)
plt.show()


# In[16]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(0)
x = np.random.choice(np.arange(100), 100)
y = np.random.choice(np.arange(100), 100)

# 마커 종류는 사각형, 색은 검은색으로 설정하여 산포도를 작성하세요
plt.scatter(x, y, marker="s", color="k") 
plt.show()


# In[17]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(0)
x = np.random.choice(np.arange(100), 100)
y = np.random.choice(np.arange(100), 100)
z = np.random.choice(np.arange(100), 100)

# z 값에 의해 마커 크기가 변하도록 플롯하세요
plt.scatter(x, y, s = z)
plt.show()


# In[18]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(0)
x = np.random.choice(np.arange(100), 100)
y = np.random.choice(np.arange(100), 100)
z = np.random.choice(np.arange(100), 100)

# z값에 따라 마커의 농도가 파란색 계열로 변하도록 설정하세요
plt.scatter(x, y, c=z, cmap="Blues") 

plt.show()


# In[19]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(0)
x = np.random.choice(np.arange(100), 100)
y = np.random.choice(np.arange(100), 100)
z = np.random.choice(np.arange(100), 100)

# z값에 따라 마커의 농도가 파란색 계열로 변하도록 설정하세요
plt.scatter(x, y, c=z, cmap="Blues")

# 컬러 바를 표시하세요
plt.colorbar()
plt.show()


# In[20]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
data = [60, 20, 10, 5, 3, 2]

# data를 원 그래프로 시각화합니다
plt.pie(data)

# 타원에서 원으로 변경하세요
plt.axis("equal")
plt.show()


# In[21]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

data = [60, 20, 10, 5, 3, 2]
labels = ["Apple", "Orange", "Banana", "Pineapple", "Kiwifruit", "Strawberry"]

# data에 labels의 라벨을 붙여 원 그래프를 시각화하세요
plt.pie(data, labels=labels)

plt.axis("equal")
plt.show()


# In[22]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

data = [60, 20, 10, 5, 3, 2]
labels = ["Apple", "Orange", "Banana", "Pineapple", "Kiwifruit", "Strawberry"]
explode = [0, 0, 0.1, 0, 0, 0]

# data에 labels의 라벨을 붙이고, Banana를 돋보이게 하여 원 그래프를 그리세요
plt.pie(data, labels=labels, explode=explode)
plt.axis("equal")
plt.show()


# In[23]:


import numpy as np
import matplotlib.pyplot as plt 

# 3D 렌더링에 필요한 라이브러리입니다
from mpl_toolkits.mplot3d import Axes3D
# get_ipython().run_line_magic('matplotlib', 'inline')

t = np.linspace(-2*np.pi, 2*np.pi)
X, Y = np.meshgrid(t, t)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Figure 오브젝트를 만듭니다
fig = plt.figure(figsize=(6,6)) 

# 서브 플롯 ax를 작성하세요
ax = fig.add_subplot(1, 1, 1, projection="3d")

# 플롯합니다
ax.plot_surface(X, Y, Z)
plt.show()


# In[25]:


# 3D 렌더링에 필요한 라이브러리입니다
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')

x = y = np.linspace(-5, 5)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2)/2) / (2*np.pi)

# Figure 오브젝트를 만듭니다
fig = plt.figure(figsize=(6, 6))

# 서브 플롯 ax를 만듭니다
ax = fig.add_subplot(1, 1, 1, projection="3d")

# 곡면을 그려주세요
ax.plot_surface(X, Y, Z) 


# In[26]:


import matplotlib.pyplot as plt
import numpy as np

# 3D 렌더링에 필요한 라이브러리입니다
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')

# Figure 오브젝트를 만듭니다
fig = plt.figure(figsize=(5, 5))

# 서브 플롯 ax1을 만듭니다
ax = fig.add_subplot(111, projection="3d")

# x, y, z의 위치를 결정합니다
xpos = [i for i in range(10)]
ypos = [i for i in range(10)]
zpos = np.zeros(10)

# x, y, z의 증가량을 결정합니다
dx = np.ones(10)
dy = np.ones(10)
dz = [i for i in range(10)]

# 3D 히스토그램을 작성하세요
ax.bar3d(xpos, ypos, zpos, dx, dy, dz) 
plt.show()


# In[27]:


import numpy as np
import matplotlib.pyplot as plt

# 3D 렌더링에 필요한 라이브러리입니다
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(0)
get_ipython().run_line_magic('matplotlib', 'inline')

X = np.random.randn(1000)
Y = np.random.randn(1000)
Z = np.random.randn(1000)

# Figure 오브젝트를 만듭니다
fig = plt.figure(figsize=(6, 6))

# 서브 플롯 ax를 만듭니다
ax = fig.add_subplot(1, 1, 1, projection="3d")

# X, Y, Z를 1차원으로 변환합니다
x = np.ravel(X)
y = np.ravel(Y)
z = np.ravel(Z)

# 3D 산포도를 작성하세요
ax.scatter3D(x, y, z) 
plt.show()


# In[28]:


import numpy as np
import matplotlib.pyplot as plt

# 컬러 맵을 위한 라이브러리입니다
from matplotlib import cm

# 3D 렌더링에 필요한 라이브러리입니다
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')

t = np.linspace(-2*np.pi, 2*np.pi)
X, Y = np.meshgrid(t, t)

R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Figure 오브젝트를 만듭니다
fig = plt.figure(figsize=(6, 6))

# 서브 플롯 ax를 만듭니다
ax = fig.add_subplot(1,1,1, projection="3d")

# 다음을 변경하여 Z값에 컬러 맵을 적용하세요
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm) 
plt.show()


# In[29]:


import matplotlib.pyplot as plt
import pandas as pd

# iris 데이터를 가져옵니다
df_iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
df_iris.columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]
fig = plt.figure(figsize=(10,10))

# setosa의 sepal length - sepal width의 관계도를 그리세요
# 라벨은 setosa, 색상은 black을 지정하세요
plt.scatter(df_iris.iloc[:50,0], df_iris.iloc[:50,1], label="setosa", color="k")

# versicolor의 sepal length - sepal width의 관계도를 그리세요
# 라벨은 versicolor, 색상은 blue를 지정하세요
plt.scatter(df_iris.iloc[50:100,0], df_iris.iloc[50:100,1], label="versicolor", color="b")

# virginica의 sepal length - sepal width의 관계도를 그리세요
# 라벨은 virginica, 색상은 green을 지정하세요
plt.scatter(df_iris.iloc[100:150,0], df_iris.iloc[100:150,1], label="virginica", color="g")

# x축의 이름을 sepal length로 하세요
plt.xlabel("sepal length")

# y축의 이름을 sepal width로 하세요
plt.ylabel("sepal width") 

# 그림을 표시합니다
plt.legend(loc="best")
plt.grid(True)
plt.show()


# In[32]:


import matplotlib.pyplot as plt
import numpy as np
import math
import time
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(100) 

X = 0 # 맞은 횟수입니다

# 시도 횟수 N을 지정하세요
N = 1000

# 사분면 경계의 방정식 [y= √ (1-x^2) (0<=x<=1)] 을 그립니다 
circle_x = np.arange(0, 1, 0.001)
circle_y = np.sqrt(1- circle_x * circle_x)
plt.figure(figsize=(5,5))
plt.plot(circle_x, circle_y) 

# N 번 시도에 걸리는 시간을 측정합니다
start_time = time.clock()

# N번 시도합니다
for i in range(0, N) :

    # 0에서 1 사이의 균일 난수를 생성하고 변수 score_x에 저장하세요
    score_x = np.random.rand()

    # 0에서 1 사이의 균일 난수를 생성하고 변수 score_y에 저장하세요
    score_y = np.random.rand()

    # 점이 원 안에 들어간 것과 들어가지 않은 경우를 조건 분기하세요
    if score_x * score_x + score_y * score_y < 1:

        # 원 안에 들어가면 검은 색으로, 들어가지 않으면 파란색으로 표시하세요
        plt.scatter(score_x, score_y, marker='o', color='k')

        # 원 안에 들어갔다면 위에서 정의한 변수 X에 1포인트를 더하세요
        X = X + 1
    else:
        plt.scatter(score_x, score_y, marker='o', color='b')
        
# pi의 근사값을 계산합니다
pi = 4*float(X)/float(N)

# 몬테카를로 법의 실행 시간을 계산합니다
end_time = time.clock()
time = end_time - start_time 

# 원주율의 결과를 표시합니다
print("원주율:%.6f"% pi)
print("실행 시간:%f" % (time)) 

# 그림을 표시합니다
plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# In[ ]:




