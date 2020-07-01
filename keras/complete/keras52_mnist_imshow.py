import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

print(f"x_train[59999]:{x_train[0]}")
print(f"y_train[59999]:{y_train[0]}")

print(f"x_train.shape:{x_train.shape}")
print(f"y_train.shape:{y_train.shape}")
print(f"x_test.shape:{x_test.shape}")
print(f"y_test.shape:{y_test.shape}")


plt.imshow(x_train[0],"gray")#imshow는 무엇인가.
# plt.show()

x_train = x_train[0] 
x_test = x_test[0]

#데이터 전처리

# from sklearn.preprocessing import OneHotEncoder
from keras.utils import np_utils
# enc = OneHotEncoder()

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# y_train=enc.fit_transform(y_train)
# x_train = x_train.reshape(-1,1)

print(f"type(x_train[0]):{type(x_train[0])}")
print(f"x.shape:{x_train.shape}")

print(f"type(x_train):{type(x_train)}")

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

x_train= x_train /255
print(f"x_train:{x_train}")


#모델구성
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten

model= Sequential()

model.add(Conv2D(10,(2,2),input_shape=(28,28,1)))
model.add(Conv2D(10,(2,2)))
model.add(Conv2D(10,(2,2)))

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense())