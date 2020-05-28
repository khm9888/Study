#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# 시드를 설정하지 않았을 때 난수가 일치하는지 확인합니다
# X, Y에 각각 다섯 개의 난수를 저장합니다
X = np.random.randn(5)
Y = np.random.randn(5)

# X, Y 값을 출력합니다
print("시드를 설정하지 않았을 때")
print("X:",X)
print("Y:",Y)

# 시드를 설정합니다
np.random.seed(0)

# 난수열을 변수에 대입합니다
x = np.random.randn(5)

# 동일한 시드를 설정하여 초기화합니다
np.random.seed(0)

# 다시 난수열을 만들여 다른 변수에 대입합니다
y = np.random.randn(5)

# x, y의 값이 일치하는지 확인합니다
print("시드를 설정했을 때")
print("x:",x)
print("y:",y)


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# 시드값을 0으로 설정하세요
np.random.seed(0)

# 정규 분포를 따르는 난수를 10,000개 생성하여 변수 x에 대입하세요
x = np.random.randn(10000)

# 시각화합니다
plt.hist(x, bins='auto')
plt.show()


# In[4]:


import numpy as np

# 시드를 설정합니다
np.random.seed(0)

# 성공 확률 0.5로 100번 시도했을 때, 성공한 횟수를 구하는 실험을 10,000회 반복하여 변수 nums에 대입하세요
nums = np.random.binomial(100, 0.5, size=10000)

# nums의 성공 횟수에서 평균을 출력하세요
print(nums.mean()/100)


# In[5]:


import numpy as np

x = ['Apple', 'Orange', 'Banana', 'Pineapple', 'Kiwifruit', 'Strawberry']

# 시드를 설정합니다
np.random.seed(0)

# x 중에서 무작위로 5개 선택하여 y에 대입합니다
y = np.random.choice(x, 5)

print(y)


# In[6]:


import datetime as dt

# 1992년 10월 22일을 나타내는 datetime 오브젝트를 생성하여 x에 할당하세요
x = dt.datetime(1992, 10, 22)

# 출력합니다
print(x)


# In[7]:


import datetime as dt

# 한 시간 반을 나타내는 timedelta 개체를 만들어 x에 대입하세요
x = dt.timedelta(hours=1, minutes=30)

# 출력합니다
print(x)


# In[8]:


import datetime as dt

# 1992년 10월 22일을 나타내는 datetime 오브젝트를 생성하여 x에 할당하세요
x = dt.datetime(1992, 10, 22)

# x에서 1일 후를 나타내는 datetime 오브젝트를 y에 대입하세요
y = x + dt.timedelta(1)

# 출력합니다
print(y)


# In[9]:


import datetime as dt

# 1992년 10월 22일을 나타내는 문자열을 "년-월-일" 형식으로 s에 대입하세요 
s = "1992-10-22"

# s를 변환하여 1992년 10월 22일을 나타내는 datetime 오브젝트를 x에 할당하세요
x = dt.datetime.strptime(s, "%Y-%m-%d")

# 출력합니다
print(x)


# In[11]:


# 문자열을 대입합니다
x = '64'
y = '16'

# x, y를 int()를 사용하여 변환하고, x,y의 합을 z에 대입합니다
z = int(x) + int(y)

# z를 출력합니다
print(z)


# In[12]:


import numpy as np

# x에 0부터 10까지의 짝수열을 대입하세요
x = np.arange(0, 11, 2)

# 출력합니다
print(x)


# In[13]:


import numpy as np

# 0에서 10까지 5항목을 동일한 간격으로 나누어 x에 대입하세요
x = np.linspace(0, 10, 5)

# 출력합니다
print(x)


# In[18]:


import matplotlib.pyplot as plt
import numpy as np

np.random.seed(100)

# 균일 난수를 10,000개 생성하여 random_number_1에 대입하세요
random_number_1 = np.random.rand(10000)

# 정규 분포를 따르는 난수를 10,000개 생성하여 random_number_2에 대입하세요
random_number_2 = np.random.randn(10000)

# 이항 분포를 따르는 난수를 10,000개 생성하여 random_number_3에 대입하세요. 성공 확률은 0.5로 하세요
random_number_3 = np.random.binomial(100, 0.5, size=(10000))

plt.figure(figsize=(5,5))

# 균일 난수를 히스토그램으로 표시합니다. bins는 50을 지정하세요
plt.hist(random_number_1, bins=50)

plt.title('uniform_distribution')
plt.grid(True)
plt.show()

plt.figure(figsize=(5,5))

# 정규 분포를 따르는 난수를 히스토그램으로 표시합니다. bins는 50을 지정하세요 
plt.hist(random_number_2, bins=50)

plt.title('normal_distribution')
plt.grid(True)
plt.show()

plt.figure(figsize=(5,5))

# 이항 분포를 따르는 난수를 히스토그램으로 표시합니다. bins는 50을 지정하세요
plt.hist(random_number_3, bins=50)

plt.title('binomial_distribution')
plt.grid(True)
plt.show()


# In[ ]:




