import matplotlib.pyplot as plt

# x=list(range(10))
# y=list(range(0,20,2))

# plt.ylim(0,10)#y값 limit

# plt.xlim([1,5])#x값 limit
# # plt.xlim(0,8)
# plt.grid()
# plt.plot(x,y)

# plt.xticks([0,1,2,3],["one","t","t","t"])

# plt.show()


# import numpy as np

# data = {'a': np.arange(50),#0~49 array
#         'c': np.random.randint(51, 101, 50),#0,50
#         'd': np.random.randn(50)}#랜덤, 정규분포 50개값
# data['b'] = data['a'] + 10 * np.random.randn(50)#a(0~49)+정규분포 10의 값 50개 값
# data['d'] = np.abs(data['d']) * 100 #정규분포 100배 (-100~100)

# plt.scatter('a', 'b', c='c', s='d', data=data)
# plt.xlabel('entry a')
# plt.ylabel('entry b')

# plt.colorbar()

# plt.show()


# plt.
# p=plt.figure()

# p.add_subplot(211)
# p.add_subplot(212)

# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# # get_ipython().run_line_magic('matplotlib', 'inline')

# np.random.seed(0)
# data = np.random.randn(10000)

# # 정규화된, 구간 수 100인 히스토그램을 작성하세요
# plt.hist(data, bins=20, normed=True,cumulative=True)
# plt.show()


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
S
# 서브 플롯 ax를 작성하세요
ax = fig.add_subplot(1, 1, 1, projection="3d")

# 플롯합니다
ax.plot_surface(X, Y, Z)
plt.show()