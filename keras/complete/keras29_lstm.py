
# x=array([range(1,4),range(2,5),range(3,6),range(4,7)]) #(4,3) 2d
# y=array(range(4,8)) #(4,) - 벡터 [4,5,6,7]
# print(y.shape)
# y2=array([range(4,8)]) #(1,4) - 벡터
# print(y2.shape)
y3=array([[4],[5],[6],[7]]) #(4,1) - 벡터
print(y3.shape)

#RNN-LSTM_connect

#데이터 1~5,2~6,3~7을 넣으면 6,7,8이 출력되도록 할 것

import numpy as np

x_train=np.array([range(1,4),range(2,5),range(3,6)])#(3,3)
y_train=np.array(range(4,7))#(3,)

x_test=np.array([range(11,14),range(12,15),range(13,16)])
y_test=np.array(range(14,17))#(3,)

print(x_train.shape)
print(y_train.shape)

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)

x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

#모델 구성

from keras.models import Model
from keras.layers import Input,Dense,LSTM,SimpleRNN

input1=Input(shape=(3,1))
dense1=LSTM(10,activation="relu",return_sequences=True)(input1)
dense2=LSTM(100)(dense1)
dense3=Dense(200)(dense2)
dense4=Dense(100)(dense3)
dense5=Dense(10)(dense4)
dense5=Dense(10)(dense5)
dense5=Dense(10)(dense5)
output1=Dense(1)(dense5)

model=Model(inputs=input1,outputs=output1)

model.summary()

#트레이닝
from keras.callbacks import EarlyStopping

model.compile(loss="mse", optimizer="adam",metrics=["accuracy"])

early=EarlyStopping(monitor="loss", patience=15)
model.fit(x_train,y_train,batch_size=1,epochs=300,verbose=0,callbacks=[early])
# model.fit(x_train,y_train,batch_size=1,epochs=100,verbose=3)

#테스트

x_pre=np.array([range(5,8)])
x_pre=x_pre.reshape(x_pre.shape[0],x_pre.shape[1],1)

print(x_test[0])

y_pre=model.predict(x_test)

mse,acc=model.evaluate(x_test,y_test,batch_size=1)

print(f"mse:{mse}, acc:{acc}")

# print(f"y_pre:{y_pre}")

# from sklearn.metrics import mean_squared_error as mse,r2_score   

# def rmse(y_test,y_pre):
#     return np.sqrt(y_test,y_pre)


# print(f"y_test:{y_test.shape}")
# print(f"y_pre:{y_pre.shape}")

# print(f"rmse:{rmse(y_test,y_pre)}")
# print(f"r2:{r2_score(y_test[0],y_pre)}")