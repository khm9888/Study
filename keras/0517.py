#RNN

#1)데이터 입력

import numpy as np

x_train=np.array([range(1,6),range(2,7),range(3,8)])
y_train=np.array(range(6,9))

print(x_train.shape)
print(y_train.shape)
    
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)

print(x_train.shape)

#2)모델 구성

from keras.models import Model
from keras.layers import Dense,Input,SimpleRNN

#함수적 모델
input1=Input(shape=(5,1))
dense1=SimpleRNN(3,activation="relu")(input1)
dense2=Dense(3)(dense1)
dense3=Dense(3)(dense2)
output1=Dense(1)(dense3)

model=Model(inputs=input1,outputs=output1)

model.summary()

#3)트레이닝

model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.fit(x_train,y_train,epochs=100,batch_size=1)

#4)테스트

x_pre=np.array([range(5,10)])
print(x_pre.shape)
x_pre=x_pre.reshape(x_pre.shape[0],x_pre.shape[1],1)
print(x_pre.shape)

mse,acc=model.evaluate(x_train,y_train,batch_size=1)
print(f"mse:{mse}")
print(f"acc:{acc}")
y_pre=model.predict(x_pre)

print(f"y_pre:{y_pre}")
