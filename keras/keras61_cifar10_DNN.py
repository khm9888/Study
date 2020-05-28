import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import Flatten, MaxPooling2D, Input
from keras.datasets import cifar10

#데이터구성
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train[0])
print('y_train[0] : ', y_train[0])

print(f"x_train.shape:{x_train.shape}")
print(f"y_train.shape:{y_train.shape}")



#minmxasclaer 적용
x_train=x_train.reshape(-1,x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
x_test=x_test.reshape(-1,x_test.shape[1]*x_test.shape[2]*x_test.shape[3])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

x_train=scaler.fit_transform(x_train)#scaler를 통해서 255로 나눔
x_test=scaler.transform(x_test)

#y값에 np_utils.to_categorical()

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


#모델

input1=Input(shape=(32*32*3,))
dense=Dense(15,activation="relu")(input1)
dense=Dense(100,activation="relu")(dense)
dense=Dense(1000,activation="relu")(dense)
dense=Dense(1000,activation="relu")(dense)
dense=Dense(100,activation="relu")(dense)
dense=Dense(15,activation="relu")(dense)
dense=Dense(10,activation="softmax")(dense)


model = Model(inputs=input1,outputs=dense)

model.summary()

#트레이닝

model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])
model.fit(x_train,y_train,batch_size=100,epochs=20,validation_split=0.3)

#테스트

loss,acc = model.evaluate(x_test,y_test,batch_size=100)

y_pre=model.predict(x_test)

y_test=np.argmax(y_test,axis=-1)
y_pre=np.argmax(y_pre,axis=-1)

print("keras61_cifar10_DNN")
print(f"loss:{loss}")
print(f"acc:{acc}")
# print(f"x_test.shape:{x_test.shape}")
# print(f"y_pre.shape:{y_pre.shape}")

print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")
