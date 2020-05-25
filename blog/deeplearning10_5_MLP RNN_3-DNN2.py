#part10 MLP 

#MLP(Multi Layer Perceptron)

#5)DNN (다입력 다:1)-> DNN

#1.데이터 구성

import numpy as np

dataset = np.array([range(1,11),range(11,21),range(21,31)])#3행 10열
dataset=np.transpose(dataset)#(10,3)


def split_5(dataset,time_steps,y_column):
    x_total=list()
    y_total=list()
    for i in range(len(dataset)-time_steps-y_column+2):#10-3-1+2
        x=dataset[i:i+time_steps,:-1]
        y=dataset[i+time_steps-1:i+time_steps-1+y_column,-1]
        x_total.append(x)
        y_total.append(y)
    return np.array(x_total),np.array(y_total)

x,y=split_5(dataset,3,1)

print(f"x:{x}")
print(f"y:{y}")

print(f"x.shape:{x.shape}")#x.shape:(8, 6, 1)
print(f"y.shape:{y.shape}")#y.shape:(8, 1)

# y=y.reshape(y.shape[0])
x=x.reshape(x.shape[0],x.shape[1]*x.shape[2])

print(f"x:{x}")
print(f"y:{y}")

print(f"x.shape:{x.shape}")#x.shape:(8, 6)
print(f"y.shape:{y.shape}")#y.shape:(8, 1)


#모델 구성

# from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()

model.add(Dense(100,input_shape=(6,),activation="relu"))
model.add(Dense(1))

# model.summary()

model.compile(loss="mse",optimizer="adam")
model.fit(x,y,epochs=1,batch_size=1)

mse=model.evaluate(x,y)
print(f"mse:{mse}")
x_pre=np.array([[9,10,11],[19,20,21]])
print(x_pre.shape[0])
x_pre=x_pre.reshape(1,x_pre.shape[0]*x_pre.shape[1])
print(f"x_pre.shape:{x_pre.shape}")

y_pre=model.predict(x_pre)

print(f"y_pe:{y_pre}")

