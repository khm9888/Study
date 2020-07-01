#44

import numpy as np
from keras.models import Model,Sequential
from keras.layers import Dense, LSTM, Input

dataset = range(1,11)
size=4
#LSTM 모델을 완성하시오

#데이터구성

def split(dataset,time_steps,y_column):
    x_list=[]
    y_list=[]
    for i in range(len(dataset)-time_steps-y_column+1):#10-5-1+1=5
        x_list.append(dataset[i:i+time_steps])
        y_list.append(dataset[i+time_steps:i+time_steps+1])
    return np.array(x_list),np.array(y_list)

x,y=split(dataset,size,1)
print(x.shape)
print(y.shape)

x=x.reshape(x.shape[0],x.shape[1],1)#6,4,1=24

y=y.reshape(x.shape[0],1)

print(x.shape)
print(y.shape)

#모델
model=Sequential()

model.add(LSTM(5,input_shape=(4,1), activation="relu"))
model.add(Dense(5))
model.add(Dense(10))
# input1 = Input(shape=(4,1))
# lstm = LSTM(5,activation="relu")(input1)
# lstm = Dense(5)(lstm)
# lstm = Dense(1)(lstm)

# model=Model(inputs=input1,outputs=lstm)

print(".\/dir")#폴더 생성해야함
print("clear")

model.summary()

