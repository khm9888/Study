#42번을 카피하여, Dense로 바꿔라

#40

import numpy as np
from keras.models import Model
from keras.layers import Dense, LSTM, Input

dataset = range(1,101)
size=4

#1)train, test 나누기

#2)마지막 6개의 행을 predict로 한다.

#3)검증(valid)의 퍼센트는 20프로


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

# x=x.reshape(x.shape[0],x.shape[1],1)#96,4,1=24
# y=y.reshape(x.shape[0],1)#96,1

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split as tts

x,x_pre,y,y_pre=tts(x,y,train_size=90/96,shuffle=False)

x_train,x_test,y_train,y_train=tts(x,y,train_size=0.9)

print(x.shape)
print(x_pre.shape)
print(x_train.shape)
print(x_test.shape)


#모델

input1 = Input(shape=(4,))
lstm = Dense(100,activation="relu")(input1)
lstm = Dense(5)(lstm)
lstm = Dense(1)(lstm)

model=Model(inputs=input1,outputs=lstm)

model.summary()

#훈련
from keras.callbacks import EarlyStopping
early = EarlyStopping(monitor="mse",patience=10,mode="min")
model.compile(loss="mse",optimizer="adam",metrics=["accuracy"])
model.fit(x,y,batch_size=1,epochs=100,callbacks=[early],validation_split=)

#테스트
loss,acc = model.evaluate(x,y,batch_size=1,)
print(f"loss:{loss}")
print(f"acc:{acc}")

y_predict=model.predict(x_pre)

print(f"y_pre:{y_predict}") 