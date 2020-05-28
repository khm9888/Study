# import numpy as np
# import random

# # x=np.arange(1,4)+1
# # print(x)

# # y=list(range(10))

# # # y[1:3]=1
# # print(y)

# # # y=x.copy()
# # y2=y.copy()
# # print(y)


# #p196 . 부울인덱스

# print(np.array([True,True,True,True]))

# arr=np.array([2,4,6,7])

# arr[1:]

# print(arr)

# print(arr%3==1)

# print("1",np.abs(range(-9,9)))

# arr=np.abs(range(-9,9))
# print(np.exp(arr))
# print(np.sqrt(arr))

# arr=np.random.randint(1, 3,(2,3))

# print(arr[1,2])

# arr=np.arange(1,10)
# arr=arr.reshape(3,3)
# print(arr.sum(axis=1))

# arr=arr.T

# print(arr)


# arr = np.array([15, 30, 5])
# arr.argsort()
# print(arr.argsort())

# print(np.sort(arr))

# np.linalg.norm(arr)

# arr=np.array([[1,0],[0,1]])
# print(arr)
# print(np.dot(arr,arr))

###################################연습문제###########################

import numpy as np

arr=np.random.randint(0, 31,(5,3))

print(arr)#0~30 사이의 5,3 배열

arr=arr.T

print(arr)#전치

arr1=arr[:,1:4]

print(arr1)#2~4열(인덱스로는 1~3열 뽑기)

arr1=np.sort(arr1,axis=1)

print(arr1)

print(np.average(arr1,axis=-1))

print(np.exp(np.arange(1,5)))

print(2.718**2)