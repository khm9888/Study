#!/usr/bin/env python
# coding: utf-8

# In[1]:


# matplotlib.pyplot을 plt로 import하세요
import matplotlib.pyplot as plt
import numpy as np
# get_ipython().run_line_magic('matplotlib', 'inline')

# np.pi는 원주율(파이)을 나타냅니다
x = np.linspace(0, 2*np.pi)
y = np.sin(x)

# 데이터 x, y를 그래프로 표시하세요
plt.plot(x,y)

plt.show()


# In[3]:


# matplotlib.pyplot을 plt로 import합니다
import matplotlib.pyplot as plt
import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')

# np.pi는 원주율(파이)을 나타냅니다
x = np.linspace(0, 2*np.pi)
y = np.sin(x)

# y축의 표시 범위를 [0,1]로 하세요
plt.ylim([0, 1])

# 데이터 x, y를 그래프에 플롯합니다
plt.plot(x, y)
plt.show()


# In[3]:


# matplotlib.pyplot을 plt로 import합니다
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')

x = np.linspace(0, 2*np.pi)
y = np.sin(x)

# 차트 제목을 설정합니다
# plt.title("y=sin(x)( 0< y< 1)")
plt.title("y=sin(x)(0<y<1)")

# 그래프의 x축과 y축의 이름을 설정하세요
plt.xlabel("x-axis")
plt.ylabel("y-axis")

# y축의 표시 범위를 [0,1]로 지정합니다
plt.ylim([0, 1])

# 데이터 x, y를 그래프에 표시합니다
plt.plot(x, y)
plt.show()


# In[6]:


# matplotlib.pyplot을 plt로 import합니다
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
x = np.linspace(0, 2 * np.pi)
y = np.sin(x)

# 차트 제목을 설정합니다
plt.title("y = sin(x)")

# 그래프의 x축과 y축의 이름을 설정합니다
plt.xlabel("x-axis")
plt.ylabel("y-axis")

# 차트에 그리드를 표시합니다
plt.grid(True)

# 데이터 x, y를 그래프에 표시합니다
plt.plot(x, y)
plt.show()


# In[8]:


# matplotlib.pyplot을 plt로 import합니다
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

x = np.linspace(0, 2*np.pi)
y = np.sin(x)

# 차트 제목을 설정합니다
plt.title("y=sin(x)")

# 그래프의 x축과 y축에 이름을 설정합니다
plt.xlabel("x-axis")
plt.ylabel("y-axis")

# 차트에 그리드를 표시합니다
plt.grid(True)

# positions와 labels를 설정합니다
positions = [0, np.pi/2, np.pi, np.pi*3/2, np.pi*2]
labels = ["0°", "90°", "180°", "270°", "360°"]

# 그래프의 x축에 눈금을 설정하세요
plt.xticks(positions, labels)

# 데이터 x, y를 그래프에 표시합니다
plt.plot(x, y)
plt.show()


# In[9]:


# matplotlib.pyplot을 plt로 import합니다
import matplotlib.pyplot as plt
import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')

x = np.linspace(0, 2*np.pi)
y1 = np.sin(x)
y2 = np.cos(x)
labels = ["90°", "180°", "270°", "360°"]
positions = [np.pi/2, np.pi, np.pi*3/2, np.pi*2]

# 차트 제목을 설정합니다
plt.title("graphs of trigonometric functions")

# 그래프의 x축과 y축에 이름을 설정합니다
plt.xlabel("x-axis")
plt.ylabel("y-axis")

# 차트에 그리드를 표시합니다
plt.grid(True)

# 그래프의 x축에 눈금을 설정합니다
plt.xticks(positions, labels)

# 그래프에 데이터 x, y1을 검은색으로 표시합니다
plt.plot(x, y1, color="k")

# 그래프에 데이터 x, y2를 파란색으로 표시합니다 
plt.plot(x, y2, color="b") 

plt.show()


# In[10]:


# matplotlib.pyplot을 plt로 import합니다
import matplotlib.pyplot as plt
import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')

x = np.linspace(0, 2*np.pi)
y1 = np.sin(x)
y2 = np.cos(x)
labels = ["90°", "180°", "270°", "360°"]
positions = [np.pi/2, np.pi, np.pi*3/2, np.pi*2]

# 차트 제목을 설정합니다
plt.title("graphs of trigonometric functions")

# 그래프의 x축과 y축에 이름을 설정합니다
plt.xlabel("x-axis")
plt.ylabel("y-axis")

# 차트에 그리드를 표시합니다
plt.grid(True)

# 그래프의 x축에 눈금을 설정합니다
plt.xticks(positions, labels)

# 데이터 x, y1을 그래프로 플롯하고, "y=sin(x)"라는 라벨을 붙여 검은색으로 표시하세요
plt.plot(x, y1, color="k", label="y=sin(x)")

# 데이터 x, y2를 그래프로 플롯하고, "y=cos(x)"라는 라벨을 붙이고 파란색으로 표시하세요
plt.plot(x, y2, color="b", label="y=cos(x)")

# 계열의 라벨을 설정하세요
plt.legend(["y=sin(x)", "y=cos(x)"])
plt.show()


# In[10]:


# matplotlib.pyplot을 plt로 import합니다
import matplotlib.pyplot as plt
import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')

x = np.linspace(0, 2*np.pi)
y = np.sin(x)

# 그림의 크기를 설정하세요
plt.figure(figsize=(4, 4))

# 데이터 x, y를 그래프에 표시합니다
plt.plot(x, y)
plt.show()


# In[11]:


# matplotlib.pyplot을 plt로 import합니다
import matplotlib.pyplot as plt
import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')

x = np.linspace(0, 2*np.pi)
y = np.sin(x)

# Figure 오브젝트를 만듭니다
fig = plt.figure(figsize=(9, 6))

# 2×3 레이아웃에서 위에서 두번째 행, 왼쪽에서 두번째 열에 서브 플롯 오브젝트를 만드세요
ax = fig.add_subplot(2, 3, 5)

# 데이터 x, y를 그래프에 표시합니다
ax.plot(x, y)

# 차트가 어디에 추가되는지 확인하기 위해, 빈 공간을 서브 플롯으로 채웁니다
axi = []
for i in range(6):
    if i==4:
        continue
    fig.add_subplot(2, 3, i+1)
plt.show()


# In[12]:


# matplotlib.pyplot을 plt로 import합니다
import matplotlib.pyplot as plt
import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')

x = np.linspace(0, 2*np.pi)
y = np.sin(x)
labels = ["90°", "180°", "270°", "360°"]
positions = [np.pi/2, np.pi, np.pi*3/2, np.pi*2]

# Figure 오브젝트를 만듭니다
fig = plt.figure(figsize=(9, 6))

# 2×3의 레이아웃에서 위에서 두번째 행, 왼쪽에서 두번째 열에 서브 플롯 객체 ax를 만듭니다
ax = fig.add_subplot(2, 3, 5)

# 그림 내 서브 플롯의 간격을 가로 세로 모두 1로 설정하세요
plt.subplots_adjust(wspace=1, hspace=1)

# 데이터 x, y를 그래프로 표시합니다
ax.plot(x, y)

# 공백을 서브 플롯으로 메웁니다
axi = []
for i in range(6):
    if i==4:
        continue
    fig.add_subplot(2, 3, i+1)
plt.show()


# In[13]:


# matplotlib.pyplot을 plt로 import합니다
import matplotlib.pyplot as plt
import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')

x = np.linspace(0, 2*np.pi)
y = np.sin(x)
labels = ["90°", "180°", "270°", "360°"]
positions = [np.pi/2, np.pi, np.pi*3/2, np.pi*2]

# Figure 오브젝트를 만듭니다
fig = plt.figure(figsize=(9, 6))

# 2×3의 레이아웃에서 위에서 두번째 행, 왼쪽에서 두번째 열에 서브 플롯 객체 ax를 만듭니다
ax = fig.add_subplot(2, 3, 5)

# 그림 내 서브 플롯의 간격을 가로 세로 모두 1로 설정하세요 
plt.subplots_adjust(wspace=1, hspace=1)

# 서브 플롯 ax 그래프의 y축 표시 범위를 [0,1]로 설정하세요
ax.set_ylim([0, 1])

# 데이터 x, y를 그래프로 표시합니다
ax.plot(x, y)

# 공백을 서브 플롯으로 메웁니다
axi = []
for i in range(6):
    if i==4:
        continue
    fig.add_subplot(2, 3, i+1)
plt.show()


# In[14]:


# matplotlib.pyplot을 plt로 import합니다
import matplotlib.pyplot as plt
import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')

x = np.linspace(0, 2*np.pi)
y = np.sin(x)
labels = ["90°", "180°", "270°", "360°"]
positions = [np.pi/2, np.pi, np.pi*3/2, np.pi*2]

# Figure 오브젝트를 만듭니다
fig = plt.figure(figsize=(9, 6))

# 2×3의 레이아웃에서 위에서 두번째 행, 왼쪽에서 두번째 열에 서브 플롯 객체 ax를 만듭니다
ax = fig.add_subplot(2, 3, 5)

# 그림 내 서브 플롯의 간격을 가로 세로 모두 1.0으로 설정하세요
plt.subplots_adjust(wspace=1.0, hspace=1.0)

# 서브 플롯 ax 그래프의 제목을 설정하세요
ax.set_title("y=sin(x)")

# 서브 플롯 ax 그래프의 x축, y축의 이름을 설정하세요
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")

# 데이터 x, y를 그래프로 표시합니다
ax.plot(x, y)

# 공백을 서브 플롯으로 메웁니다
axi = []
for i in range(6):
    if i==4:
        continue
    fig.add_subplot(2, 3, i+1)
plt.show()


# In[15]:


# matplotlib.pyplot을 plt로 import합니다
import matplotlib.pyplot as plt
import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')

x = np.linspace(0, 2*np.pi)
y = np.sin(x)

# Figure 오브젝트를 만듭니다
fig = plt.figure(figsize=(9, 6))

# 2×3의 레이아웃에서 위에서 두번째 행, 왼쪽에서 두번째 열에 서브 플롯 객체 ax를 만듭니다
ax = fig.add_subplot(2, 3, 5)

# 그림 내 서브 플롯의 간격을 가로 세로 모두 1.0으로 설정하세요
plt.subplots_adjust(wspace=1.0, hspace=1.0)

# 서브 플롯 ax의 그래프에 그리드를 설정하세요
ax.grid(True)

# 서브 플롯 ax 그래프의 제목을 설정하세요
ax.set_title("y=sin(x)")

# 서브 플롯 ax 그래프의 x축, y축의 이름을 설정하세요
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")

# 데이터 x, y를 그래프로 표시합니다
ax.plot(x, y)

# 공백을 서브 플롯으로 메웁니다
axi = []

for i in range(6):
    if i==4:
        continue
    fig.add_subplot(2, 3, i+1)
plt.show()


# In[16]:


# matplotlib.pyplot을 plt로 import합니다
import matplotlib.pyplot as plt
import numpy as np
# get_ipython().run_line_magic('matplotlib', 'inline')
x = np.linspace(0, 2*np.pi)
y = np.sin(x)
positions = [0, np.pi/2, np.pi, np.pi*3/2, np.pi*2]
labels = ["0°", "90°", "180°", "270°", "360°"]

# Figure 오브젝트를 만듭니다
fig = plt.figure(figsize=(9, 6))

# 2×3의 레이아웃에서 위에서 두번째 행, 왼쪽에서 두번째 열에 서브 플롯 객체 ax를 만듭니다
ax = fig.add_subplot(2, 3, 5)

# 그림 내 서브 플롯의 간격을 가로 세로 모두 1로 설정하세요
plt.subplots_adjust(wspace=1, hspace=1)

# 서브 플롯 ax의 그래프에 그리드를 설정하세요 
ax.grid(True)

# 서브 플롯 ax 그래프의 제목을 설정하세요
ax.set_title("y=sin(x)")

# 서브 플롯 ax 그래프의 x축, y축의 이름을 설정하세요
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")

# 서브 플롯 ax 그래프의 x축에 눈금을 설정하세요
ax.set_xticks(positions)
ax.set_xticklabels(labels) 

# 데이터 x, y를 그래프로 표시합니다
ax.plot(x, y)

# 공백을 서브 플롯으로 메웁니다
axi = []
for i in range(6):
    if i==4:
        continue
    fig.add_subplot(2, 3, i+1)
plt.show()


# In[17]:


# matplotlib.pyplot을 plt로 import합니다
import matplotlib.pyplot as plt
import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')

x_upper = np.linspace(0, 5)
x_lower = np.linspace(0, 2 * np.pi)
x_tan = np.linspace(-np.pi / 2, np.pi / 2)
positions_upper = [i for i in range(5)]
positions_lower = [0, np.pi / 2, np.pi, np.pi * 3 / 2, np.pi * 2]
positions_tan = [-np.pi / 2, 0, np.pi / 2]
labels_upper = [i for i in range(5)]
labels_lower = ["0°", "90°", "180°", "270°", "360°"]
labels_tan = ["-90°", "0°", "90°"]

# Figure 오브젝트를 만듭니다
fig = plt.figure(figsize=(9, 6))

# 3×2 레이아웃으로 여러 함수의 그래프를 플롯합니다
# 서브 플롯들이 겹치지 않도록 설정합니다
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# 상단의 서브 플롯을 만듭니다
for i in range(3):
    y_upper = x_upper ** (i + 1)
    ax = fig.add_subplot(2, 3, i + 1)

    # 서브 플롯 ax 그래프에 그리드를 표시합니다
    ax.grid(True)

    # 서브 플롯 ax 그래프의 제목을 설정합니다
    ax.set_title("$y=x^%i$" % (i + 1))

    # 서브 플롯 ax 그래프의 x축, y축에 이름을 설정합니다
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")

    # 서브 플롯 ax 그래프의 x축에 눈금을 설정합니다
    ax.set_xticks(positions_upper)
    ax.set_xticklabels(labels_upper)

    # 데이터 x, y를 그래프로 표시합니다
    ax.plot(x_upper, y_upper)

# 하단의 서브 플롯을 만듭니다
# 리스트에 사용할 함수와 제목을 미리 넣은 뒤, for 문으로 처리합니다
y_lower_list = [np.sin(x_lower), np.cos(x_lower)]
title_list = ["$y=sin(x)$", "$y=cos(x)$"]
for i in range(2):
    y_lower = y_lower_list[i]
    ax = fig.add_subplot(2, 3, i + 4)

    # 서브 플롯 ax 그래프에 그리드를 표시합니다
    ax.grid(True)

    # 서브 플롯 ax 그래프의 제목을 설정합니다
    ax.set_title(title_list[i])
    
    # 서브 플롯 ax 그래프의 x축, y축에 이름을 설정합니다
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")

    # 서브 플롯 ax 그래프의 x축에 라벨을 설정합니다
    ax.set_xticks(positions_lower)
    ax.set_xticklabels(labels_lower)

    # 데이터 x, y를 그래프로 표시합니다
    ax.plot(x_lower, y_lower)

# y=tan(x) 그래프의 플롯
ax = fig.add_subplot(2, 3, 6)

# 서브 플롯 ax 그래프에 그리드를 표시합니다
ax.grid(True)

# 서브 플롯 ax 그래프의 제목을 설정합니다
ax.set_title("$y=tan(x)$")

# 서브 플롯 ax 그래프의 x축, y축에 이름을 설정합니다
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")

# 서브 플롯 ax 그래프의 x축에 눈금을 설정합니다
ax.set_xticks(positions_tan)
ax.set_xticklabels(labels_tan)

# 서브 플롯 ax 그래프의 y축의 범위를 설정합니다
ax.set_ylim(-1, 1)

# 데이터 x, y를 그래프로 표시합니다
ax.plot(x_tan, np.tan(x_tan))

plt.show()


# In[ ]:




