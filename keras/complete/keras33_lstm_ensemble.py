import numpy as np

#데이터 구성

dataset=np.array(range(1,13))
dataset2=np.array(range(20,70,10))
dataset3=np.array(range(10,130,10))
dataset4=np.array(range(2,7))

# print(dataset2)
def split(dataset,time_steps):
    arr=list()
    for i in range(len(dataset)-time_steps+1):
        input_v=dataset[i:i+time_steps]
        # print(input_v)
        arr.append(input_v)
    return arr
x1=np.array(split(dataset,3)+split(dataset2,3))
x2=np.array(split(dataset3,3)+split(dataset4,3))

y=np.array(list(range(4,14))+list(range(50,80,10)))

print(f"x1:{x1}")
print(f"x2:{x2}")
print(f"y:{y}")
# y=np.array(list(range(4,14))+list(range(50,80,10)))
# print(y)

x1=x1.reshape(x1.shape[0],x1.shape[1],1)
x2=x2.reshape(x2.shape[0],x2.shape[1],1)

print(x1)

x_pre=np.array([55, 65, 75])
x_pre=x_pre.reshape(1,x_pre.shape[0],1)

x_pre_2=np.array([65, 75, 85])
x_pre_2=x_pre_2.reshape(1,x_pre_2.shape[0],1)
#모델

from keras.models import Model
from keras.layers import Dense, LSTM,Input,Concatenate

input1=Input(shape=(3,1))
lstm1=LSTM(10,activation="relu",input_shape=(3,1),return_sequences=False)(input1)
dense1=Dense(64)(lstm1)

dense1=Dense(1)(dense1)

input2=Input(shape=(3,1))
lstm2=LSTM(10,activation="relu",input_shape=(3,1),return_sequences=False)(input2)
dense2=Dense(64)(lstm2)
dense2=Dense(1)(dense2)

merge = Concatenate()([dense1,dense2])

merge=Dense(64)(merge)

merge=Dense(1)(merge)

model=Model(inputs=[input1,input2],outputs=merge)
# model.summary()


#트레이닝

from keras.callbacks import EarlyStopping


model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

early=EarlyStopping(monitor="loss",patience=10,mode="min")

model.fit([x1,x2],y,batch_size=1,epochs=500,verbose=1,callbacks=[early])
# model.fit(x,y,batch_size=1,epochs=100,verbose=3)

#검증

loss, acc=model.evaluate([x1,x2],y,batch_size=1)

print(f"loss:{loss}")
print(f"acc:{acc}")

y_pre=model.predict([x_pre,x_pre_2])

print(f"y_pre : {y_pre}")