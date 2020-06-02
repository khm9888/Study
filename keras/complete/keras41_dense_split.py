#40

import numpy as np
from keras.models import Model
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

# x=x.reshape(x.shape[0],x.shape[1])#6,4

# y=y.reshape(x.shape[0],1)

print(x.shape)
print(y.shape)


#모델

input1 = Input(shape=(4,))
dense = Dense(100,activation="relu")(input1)
dense = Dense(5)(dense)
dense = Dense(1)(dense)

model=Model(inputs=input1,outputs=dense)

model.summary()

#훈련
from keras.callbacks import EarlyStopping
early = EarlyStopping(monitor="mse",patience=10,mode="min")
model.compile(loss="mse",optimizer="adam",metrics=["accuracy"])
model.fit(x,y,batch_size=1,epochs=100,callbacks=[early])

#테스트
loss,acc = model.evaluate(x,y,batch_size=1,)
print(f"loss:{loss}")
print(f"acc:{acc}")

y_pre=model.predict(x)

print(f"y_pre:{y_pre}")