#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 필요한 라이브러리를 import합니다
import numpy as np
import time
from numpy.random import rand

# 행, 열의 크기
N = 150

# 행렬을 초기화합니다
matA = np.array(rand(N, N))
matB = np.array(rand(N, N))
matC = np.array([[0] * N for _ in range(N)])

# 파이썬의 리스트를 사용하여 계산합니다
# 시작 시간을 저장합니다
start = time.time()

# for 문을 사용하여 행렬 곱셈을 실행합니다
for i in range(N):
    for j in range(N):
        for k in range(N):
            matC[i][j] = matA[i][k] * matB[k][j]

print("파이썬의 기능만으로 계산한 결과：%.2f[sec]" % float(time.time() -start))

# NumPy를 사용하여 계산합니다
# 시작 시간을 저장합니다 
start = time.time()

# NumPy를 사용하여 행렬 곱셈을 실행합니다
matC = np.dot(matA, matB)

# 소수점 이하 두 자리까지 표시되므로 NumPy는 0.00[sec]로 표시됩니다
print("NumPy를 사용한 경우의 계산 결과：%.2f[sec]" % float(time.time() - start))


# In[2]:


import numpy as np

storages = [24, 3, 4, 23, 10, 12]
print(storages)

# ndarray 배열을 생성하고 변수 np_storages에 대입하세요
np_storages = np.array(storages)

# 변수 np_storages의 자료형을 출력하세요
print(type(np_storages))


# In[3]:


# NumPy를 사용하지 않고 실행합니다
storages = [1, 2, 3, 4]
new_storages = []
for n in storages:
    n += n
    new_storages.append(n)
print(new_storages)


# In[4]:


# NumPy를 사용하여 실행합니다
import numpy as np
storages = np.array([1, 2, 3, 4])
storages += storages
print(storages)


# In[6]:


import numpy as np

arr = np.array([2, 5, 3, 4, 8])

# arr + arr
print('arr + arr')
print(arr + arr)

# arr - arr
print('arr - arr')
print(arr - arr)

# arr ** 3
print('arr ** 3')
print(arr ** 3)

# 1 / arr
print('1 / arr')
print(1 / arr)


# In[7]:


arr = np.arange(10)
print(arr)


# In[8]:


arr = np.arange(10)
arr[0:3] = 1
print(arr)


# In[9]:


import numpy as np

arr = np.arange(10)
print(arr)

# 변수 arr의 요소 중 3,4,5 만 출력합니다
print(arr[3:6])

# 변수 arr의 요소 중 3,4,5을 24로 변경하십시오
arr[3:6] = 24
print(arr)


# In[2]:


import numpy as np

# ndarray를 그대로 arr2 변수에 대입한 경우를 살펴봅시다
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)

arr2 = arr1
arr2[0] = 100

# arr2 변수를 변경하면 원래 변수(arr1)도 영향을 받고 있습니다
print(arr1)

# ndarray를 copy()를 사용해 arr2 변수에 대입한 경우를 살펴봅시다
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)

arr2 = arr1.copy()
arr2[0] = 100

# arr2 변수를 변경해도 원래 변수(arr1)에 영향을 주지 않습니다
print(arr1)


# In[3]:


import numpy as np

# 파이썬의 리스트에 슬라이스를 이용한 경우를 확인합니다
arr_List = [x for x in range(10)]
print("리스트 형 데이터입니다")
print("arr_List :", arr_List)

print()
arr_List_copy = arr_List[:]
arr_List_copy[0] = 100

print("리스트의 슬라이스는 복사본이 생성되므로, arr_List에는 arr_List_copy의 변경점이 반영되지 않습니다.")
print("arr_List:",arr_List)
print()

# NumPy의 ndarray에 슬라이스를 이용한 경우를 확인합니다
arr_NumPy = np.arange(10)
print("NumPy의 ndarray 데이터입니다")
print("arr_NumPy:",arr_NumPy)
print()

arr_NumPy_view = arr_NumPy[:]
arr_NumPy_view[0] = 100

print("NumPy의 슬라이스는 view(데이터가 저장된 위치의 정보)가 대입되므로, arr_NumPy_view를 변경하면 arr_NumPy에 반영됩니다")
print("arr_NumPy:",arr_NumPy)
print()

# NumPy의 ndarray에서 copy()를 이용한 경우를 확인합니다
arr_NumPy = np.arange(10)
print("NumPy의 ndarray에서 copy()를 이용한 경우입니다")
print("arr_NumPy:",arr_NumPy)
print()
arr_NumPy_copy = arr_NumPy[:].copy()
arr_NumPy_copy[0] = 100

print("copy()를 사용하면 복사본이 생성되기 때문에 arr_NumPy_copy는 arr_NumPy에 영향을 미치지 않습니다")
print("arr_NumPy:",arr_NumPy)


# In[13]:


arr = np.array([2, 4, 6, 7])
print(arr[np.array([True, True, True, False])])


# In[14]:


arr = np.array([2, 4, 6, 7])
print(arr[arr % 3 == 1])


# In[4]:


import numpy as np

arr = np.array([2, 3, 4, 5, 6, 7])

# 변수 arr의 각 요소가 2로 나누어 떨어지는지를 나타내는 부울 배열을 출력하세요
print(arr % 2 == 0)

# 변수 arr의 각 요소 중에서 2로 나누어 떨어진 것의 배열을 출력하세요
print(arr[arr % 2 == 0])


# In[5]:


import numpy as np

arr = np.array([4, -9, 16, -4, 20])
print(arr)

# 변수 arr의 각 요소를 절대값으로 하여 변수 arr_abs에 대입하세요
arr_abs = np.abs(arr)
print(arr_abs)

# 변수 arr_abs의 각 요소의 e의 거듭제곱과 제곱근을 출력하세요
print(np.exp(arr_abs))
print(np.sqrt(arr_abs))


# In[6]:


import numpy as np

arr1 = [2, 5, 7, 9, 5, 2]
arr2 = [2, 5, 8, 3, 1]

# np.unique() 함수를 사용하여 중복을 제거한 배열을 변수 new_arr1에 대입하세요
new_arr1 = np.unique(arr1)
print(new_arr1)

# 변수 new_arr1과 변수 arr2의 합집합을 출력하세요
print(np.union1d(new_arr1, arr2))

# 변수 new_arr1과 변수 arr2의 교집합을 출력하세요
print(np.intersect1d(new_arr1, arr2))

# 변수 new_arr1에서 변수 arr2을 뺀 차집합을 출력하세요
print(np.setdiff1d(new_arr1, arr2))


# In[8]:


import numpy as np

# np.random을 적지 않아도 randint() 함수를 사용할 수 있도록 import 하세요 
from numpy.random import randint

# 변수 arr1에 각 요소가 0 이상 10 이하인 정수 행렬(5 × 2)를 대입하세요
arr1 = randint(0, 11, (5, 2))
print(arr1)

# 변수 arr2에 0 이상 1 미만의 난수를 3개 생성해 대입하세요
arr2 = np.random.rand(3)
print(arr2)


# In[9]:


import numpy as np

# 변수 arr에 2차원 배열을 대입하세요
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(arr)

# 변수 arr 행렬의 각 차원의 요소 수를 출력하세요
print(arr.shape)

# 변수 arr을 4행 2열의 행렬로 변환합니다
print(arr.reshape(4, 2))


# In[23]:


arr = np.array([[1, 2 ,3], [4, 5, 6]])
print(arr[1])


# In[24]:


arr = np.array([[1, 2 ,3], [4, 5, 6]])
print(arr[1,2])


# In[25]:


arr = np.array([[1, 2 ,3], [4, 5, 6]])
print(arr[1,1:])


# In[10]:


import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr) 

# 변수 arr의 요소 중 3을 출력하세요
print(arr[0, 2])

# 변수 arr에서 부분적으로 꺼내어 출력하세요
# "1행 이후, 2열까지"를 꺼냅니다
print(arr[1:, :2])


# In[27]:


arr = np.array([[1, 2 ,3], [4, 5, 6]])

print(arr.sum())
print(arr.sum(axis=0))
print(arr.sum(axis=1))


# In[11]:


import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 12], [15, 20, 22]])

# arr 행의 합을 구하여 문제에서 제시한 1차원 배열을 반환하세요
print(arr.sum(axis=1))


# In[13]:


arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

# 3행, 2행, 0행을 순서대로 추출하여 새로운 요소를 만듭니다
# 인덱스 번호는 0부터 시작합니다
print(arr[[3, 2, 0]])


# In[14]:


import numpy as np

arr = np.arange(25).reshape(5, 5)

# 변수 arr의 행의 순서를 변경하여 출력하세요
print(arr[[1, 3, 0]])


# In[15]:


import numpy as np

arr = np.arange(10).reshape(2, 5)

# 변수 arr을 전치하세요
print(arr.T) # print(np.transpose(arr))


# In[32]:


arr = np.array([15, 30, 5])
arr.argsort()


# In[16]:


import numpy as np

arr = np.array([[8, 4, 2], [3, 5, 1]])

# argsort() 메서드로 출력하세요
print(arr.argsort())

# np.sort() 함수로 정렬하여 출력하세요
print(np.sort(arr))

# sort() 메서드로 행을 정렬하세요
arr.sort(1)
print(arr)


# In[17]:


import numpy as np

# arr을 정의합니다
arr = np.arange(9).reshape(3, 3)

# 변수 arr와 arr의 행렬곱을 출력하세요
print(np.dot(arr, arr))

# vec를 정의합니다
vec = arr.reshape(9)

# 변수 vec의 노름을 출력하세요
print(np.linalg.norm(vec))


# In[18]:


import numpy as np

arr = np.arange(15).reshape(3, 5)

# 변수 arr의 각 열의 평균을 출력하세요
print(arr.mean(axis=0))

# 변수 arr의 행 합계를 출력하세요
print(arr.sum(axis=1))

# 변수 arr의 최소값을 출력하세요
print(arr.min())

# 변수 arr의 각 열의 최대값의 인덱스 번호를 출력하세요
print(arr.argmax(axis=0))


# In[36]:


x = np.arange(6).reshape(2, 3)
print(x + 1)


# In[2]:


import numpy as np

# 0에서 14 사이의 정수값을 갖는 3 × 5의 ndarray 배열 x를 생성합니다
x = np.arange(15).reshape(3, 5)
print("x:")
print(x)

# 0에서 4 사이의 정수값을 갖는 1 × 5의 ndarray 배열 y를 생성합니다
y = np.array([np.arange(5)])
print("y:")
print(y)

# x의 n번째 열의 모든 행에서 n만 빼세요
z = x - y

# x를 출력
print("z:")
print(z)


# In[20]:


import numpy as np

np.random.seed(100)

# 각 요소가 0 ~ 30인 정수 행렬(5 × 3)을 변수 arr에 대입하세요
arr = np.random.randint(0, 31, (5, 3))
print(arr)

# 변수 arr을 전치하세요
arr = arr.T
print(arr)

# 변수 arr의 2, 3, 4열만 추출한 행렬(3 × 3)을 변수 arr1에 대입하세요
arr1 = arr[:, 1:4]
print(arr1)

# 변수 arr1의 행을 정렬하세요
arr1.sort(0)
print(arr1)

# 각 열의 평균을 출력하세요
print(arr1.mean(axis = 0))


# In[25]:


import numpy as np

# 난수를 초기화합니다
np.random.seed(0)

# 가로세로 크기를 전달하면 해당 크기의 이미지를 난수로 채워 생성하는 함수입니다
def make_image(m, n) :

    # n × m 행렬의 각 성분을 0~5의 난수로 채웁니다
    image = np.random.randint(0, 6, (m, n)) 
    return image

# 전달된 행렬의 일부를 변경하는 함수입니다
def change_little(matrix) :
    # 전달받은 행렬의 형태(크기)를 취득하여 shape에 대입하세요
    shape = matrix.shape
    # 행렬의 각 성분에 대해, 변경 여부를 무작위로 결정한 다음
    # 변경할 때는 0-5 사이의 정수로 임의로 교체하세요
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.random.randint(0, 2)==1:
                matrix[i][j] = np.random.randint(0, 6, 1)
    return matrix

# 임의의 이미지를 만듭니다
image1 = make_image(3, 3)
print(image1)
print()

# 임의의 변경 사항을 적용합니다
image2 = change_little(np.copy(image1))
print(image2)
print()

# image1과 image2의 차이를 계산하여 image3에 대입하세요
image3 = image2 - image1
print(image3)
print()

# image3의 각 성분을 절대값으로 한 행렬을 image3에 다시 대입하세요
image3 = np.abs(image3)

# image3을 출력합니다
print(image3)


# In[ ]:




