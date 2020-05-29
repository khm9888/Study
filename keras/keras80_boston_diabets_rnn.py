import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import Flatten, MaxPool2D, Input,LSTM
from sklearn.datasets import load_diabetes
from keras.utils import np_utils

#데이터구성
dataset = load_diabetes()
x=dataset.data
y=dataset.target

#dimention 확인
print(f"x.shape:{x.shape}")
print(f"y.shape:{y.shape}")

print(f"x[0]:{x[0]}")
print(f"y[0]:{y[0]}")

print(f"x:{x}")
print(f"y:{y}")

#2차원이라 무의미하다.

# x_train=x_train.reshape(-1,x_train.shape[1])
# x_test=x_test.reshape(-1,x_test.shape[1])


from keras.utils import np_utils
y = np_utils.to_categorical(y)

#분리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x=scaler.fit_transform(x)#scaler를 통해서 255로 나눔

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,train_size=0.9)

# print(f"x_train[0]:{x_train[0]}")
# print(f"y_train[0]:{y_train[0]}")
x_train=x_train.reshape(-1,x_train.shape[1],1)
x_test=x_test.reshape(-1,x_test.shape[1],1)

#모델

input1=Input(shape=(4,1))
dense=LSTM(3000,activation="relu")(input1)
dense=Dense(3,activation="softmax")(dense)

model = Model(inputs=input1,outputs=dense)

model.summary()

#트레이닝

model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])
model.fit(x_train,y_train,batch_size=30,epochs=20,validation_split=0.3)

#테스트

loss,acc = model.evaluate(x_test,y_test,batch_size=100)

y_pre=model.predict(x_test)

y_test=np.argmax(y_test,axis=-1)
y_pre=np.argmax(y_pre,axis=-1)
print("keras79_boston_diabets_dnn")
print(f"loss:{loss}")
print(f"acc:{acc}")
# print(f"x_test.shape:{x_test.shape}")
# print(f"y_pre.shape:{y_pre.shape}")

print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")

#keras79_boston_diabets_dnn
"""
keras79_boston_diabets_dnn
loss:0.025108538568019867
acc:0.9971181154251099
y_test[0:20]:[ 55 173 232 101 153 173 154  64 178 281 143 268  83 142  68 262 245 262
  84 190]
y_pre[0:20]:[ 72 109 263  48 109 109 109 109 131 109 109 281 104 116 200 229 310 109
  87 109]
"""
