#part10 MLP 

#MLP(Multi Layer Perceptron)

# 9)RNN (다입력 다:다)-두번째 ->DNN

#1.데이터 구성

import numpy as np

dataset = np.array([range(1,11),range(11,21),range(21,31)])#3행 10열
dataset=np.transpose(dataset)#(10,3)


def split_9(dataset,time_steps,y_column):#(dataset,3,1)
    x_total=list()
    y_total=list()
    for i in range(len(dataset)-time_steps-y_column+1):#10-3-1+1
        x=dataset[i:i+time_steps]#3번째 행
        y=dataset[i+time_steps:i+time_steps+y_column]#4번째 행 이후
        x_total.append(x)
        y_total.append(y)
    return np.array(x_total),np.array(y_total)

x,y=split_9(dataset,3,1)

print(f"x:{x}")
print(f"y:{y}")

print(f"x.shape:{x.shape}")#(7,3,3)
print(f"y.shape:{y.shape}")#(7,1,3)

x=x.reshape(x.shape[0],x.shape[1]*x.shape[2])#7,9
y=y.reshape(y.shape[0],y.shape[2])#7,3
# print(f"x:{x}")
# print(f"y:{y}")

print(f"x.shape:{x.shape}")#7,9
print(f"y.shape:{y.shape}")#7,3

#모델 구성

# from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()

model.add(Dense(100,input_shape=(9,),activation="relu"))#Dense
model.add(Dense(3))

# model.summary()

model.compile(loss="mse",optimizer="adam")
model.fit(x,y,epochs=200,batch_size=1)

mse=model.evaluate(x,y)
print(f"mse:{mse}")
x_pre=np.array([[9,10,11],[19,20,21],[29,30,31]])#(3,3)
x_pre=np.transpose(x_pre)#(3,3)
x_pre=x_pre.reshape(1,x_pre.shape[0]*x_pre.shape[1])#(1,9)
# x_pre=x_pre.reshape(1,x_pre.shape[0]mx_pre.shape[1])#(1,3,3)
print(f"x_pre.shape:{x_pre.shape}")

y_pre=model.predict(x_pre)

print(f"y_pe:{y_pre}")

