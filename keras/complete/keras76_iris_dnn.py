import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import Flatten, MaxPool2D, Input
from sklearn.datasets import load_iris

#데이터구성
dataset = load_iris()

print(dataset.keys())

[print (f"{i} : {dataset[i]} \n {'-'*50}") for i in list(dataset.keys())[2:-1]]

x=dataset.data
y=dataset.target

#dimention 확인
print(f"x.shape:{x.shape}")
print(f"y.shape:{y.shape}")

# print(f"x[0]:{x[0]}")
# print(y[0])

#2차원이라 무의미하다.

# x_train=x_train.reshape(-1,x_train.shape[1])
# x_test=x_test.reshape(-1,x_test.shape[1])


#분리
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,train_size=0.9)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

x_train=scaler.fit_transform(x_train)#scaler를 통해서 255로 나눔
x_test=scaler.transform(x_test)

#y값에 np_utils.to_categorical()
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


#모델

input1=Input(shape=(4,))
dense=Dense(3000,activation="relu")(input1)
dense=Dense(3,activation="softmax")(dense)#다중분류

model = Model(inputs=input1,outputs=dense)

model.summary()

#트레이닝

model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])
model.fit(x_train,y_train,batch_size=30,epochs=50,validation_split=0.3)

#테스트

loss,acc = model.evaluate(x_test,y_test,batch_size=100)

y_pre=model.predict(x_test)

y_test=np.argmax(y_test,axis=-1)
y_pre=np.argmax(y_pre,axis=-1)
print(__file__)
print(f"loss:{loss}")
print(f"acc:{acc}")
# print(f"x_test.shape:{x_test.shape}")
# print(f"y_pre.shape:{y_pre.shape}")

print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")

#keras76_boston_iris_dnn
"""
keras76_boston_iris_dnn
loss:0.11052168160676956
acc:1.0
y_test[0:20]:[1 1 1 0 1 2 2 1 2 0 2 1 2 1 2]
y_pre[0:20]:[1 1 1 0 1 2 2 1 2 0 2 1 2 1 2]
"""
