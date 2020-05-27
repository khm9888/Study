#part10 MLP 

#MLP(Multi Layer Perceptron)

#4)DNN (다입력 다:1)

#1.데이터 구성

import numpy as np

dataset = np.array([range(1,11),range(11,21),range(21,31)])#3행 10열
dataset=np.transpose(dataset)



def split(dataset,time_steps,y_column):
    x_total=list()
    y_total=list()
    for i in range(len(dataset)-time_steps+1):#10-3+1
        x=dataset[i:i+time_steps,:-1]
        y=dataset[i+time_steps-1:i+time_steps-1+y_column,-1]
        x_total.append(x)
        y_total.append(y)
    return np.array(x_total),np.array(y_total)

x,y=split(dataset,3,1)

print(f"x:{x}")
print(f"y:{y}")

print(f"x.shape:{x.shape}")#x.shape:(8, 3, 2)

print(f"y.shape:{y.shape}")#y.shape:(8, 1)

y=y.reshape(y.shape[0])


#모델 구성

# from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()

model.add(LSTM(100,input_shape=(3,2),activation="relu"))
model.add(Dense(1))

model.summary()

model.compile(loss="mse",optimizer="adam")
model.fit(x,y,epochs=500,batch_size=1,verbose=0)

mse=model.evaluate(x,y)
print(f"mse:{mse}")
x_pre=np.array([[9,10,11],[19,20,21]])
x_pre=np.transpose(x_pre)
x_pre=x_pre.reshape(1,x_pre.shape[0],x_pre.shape[1])
print(f"x_pre.shape:{x_pre.shape}")

y_pre=model.predict(x_pre)

print(f"y_pe:{y_pre}")

