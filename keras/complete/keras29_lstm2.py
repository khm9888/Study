import numpy as np

x=np.array([range(1,4),range(2,5),range(3,6),range(4,7)])
y=np.array(range(4,8))

print(f"x.shape:{x.shape}") #4,3
print(f"y.shape:{y.shape}") #4,

x=x.reshape(x.shape[0],x.shape[1],1)

print(f"x.shape:{x.shape}") #4,3,1

# x의 shape = (batch_size, timesteps, feature)
# input_shape = (timesteps, feature)
# input_length = timesteps, input_dim = feature

from sklearn.model_selection import train_test_split as tts

x_train,x_test,y_train,y_test=tts(x,y,train_size=0.8,shuffle=False)

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