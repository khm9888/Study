#part10 MLP 

#MLP(Multi Layer Perceptron)

#3)DNN (다대다)

#1.데이터 구성

import numpy as np

dataset = np.array(range(1,11))

# x는 4개씩 자르고, y는 2개씩 자름

def split(dataset,time_steps,y_column):
    x_total=list()
    y_total=list()
    for i in range(len(dataset)-time_steps-y_column+1):
        x=dataset[i:i+time_steps]
        y=dataset[i+time_steps:i+time_steps+y_column]
        x_total.append(x)
        y_total.append(y)
    return np.array(x_total),np.array(y_total)

x,y=split(dataset,4,2)

print(f"x:{x}")
print(f"y:{y}")

print(f"x.shape:{x.shape}")#x.shape:(5, 4)
print(f"y.shape:{y.shape}")#y.shape:(5, 2)

x=x.reshape(x.shape[0],x.shape[1],1)

print(f"x.shape:{x.shape}")#x.shape:(5, 4, 1)
print(f"y.shape:{y.shape}")#y.shape:(5, 2)

#모델 구성

# from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()

model.add(LSTM(100,input_shape=(4,1),activation="relu"))
model.add(Dense(2))

model.summary()

model.compile(loss="mse",optimizer="adam")
model.fit(x,y,batch_size=1,epochs=500)

mse=model.evaluate(x,y)
print(f"mse:{mse}")
x_pre=np.array([7,8,9,10])
# print(x_pre.shape)
x_pre=x_pre.reshape(1,x_pre.shape[0],1)

y_pre=model.predict(x_pre)

print(f"y_pe:{y_pre}")

