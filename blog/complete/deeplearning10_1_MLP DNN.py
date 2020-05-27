#part10 MLP 

#MLP(Multi Layer Perceptron)

#1)DNN

#1.데이터 구성

import numpy as np

dataset = np.array(range(1,11))

# x는 4개씩 자르고, y는 1개씩 자름

def split(dataset,y_column):
    x_total=list()
    y_total=list()
    for i in range(len(dataset)-y_column):#10-4
        x=dataset[i:i+y_column]
        y=dataset[i+y_column]
        x_total.append(x)
        y_total.append(y)
    return np.array(x_total),np.array(y_total)

x,y=split(dataset,4)

print(f"x:{x}")
print(f"y:{y}")


print(f"x.shape:{x.shape}")#x.shape:(6, 4)

print(f"y.shape:{y.shape}")#y.shape:(6,)

#모델 구성

from keras.models import Model
from keras.layers import Input, Dense

input1=Input(shape=(4,))
dense1=Dense(100)(input1)
dense1=Dense(1)(dense1)

model=Model(inputs=input1,outputs=dense1)

# model.summary()

model.compile(loss="mse",optimizer="adam",metrics=["accuracy"])
model.fit(x,y,batch_size=1,epochs=500)

mse=model.evaluate(x,y)
print(f"mse:{mse}")

x_pre=np.array([7,8,9,10])
# print(x_pre.shape)
x_pre=x_pre.reshape(1,x_pre.shape[0])

y_pre=model.predict(x_pre)

print(f"y_pe:{y_pre}")

