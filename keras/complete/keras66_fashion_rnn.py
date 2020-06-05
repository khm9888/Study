import numpy as np
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import Flatten, MaxPooling2D, Input,LSTM
from keras.datasets import cifar10

#데이터구성
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train[0])
print('y_train[0] : ', y_train[0])

print(f"x_train.shape:{x_train.shape}")
print(f"y_train.shape:{y_train.shape}")

x_train=x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])
x_test=x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])

#minmxasclaer 적용
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_train=scaler.fit_transform(x_train)#scaler를 통해서 255로 나눔
x_test=scaler.transform(x_test)



#y값에 np_utils.to_categorical()

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#다시 4차원으로 변경

x_train=x_train.reshape(-1,28,28)
x_test=x_test.reshape(-1,28,28)


#모델

input1=Input(shape=(28,28))
# dense=LSTM(200,activation="relu",return_sequences=True)(input1)
dense=LSTM(200,activation="relu",return_sequences=False)(input1)
# model.add(LSTM(20,activation="relu"))

# dense=Dense(10, activation="relu")(dense)
# dense=Dense(10, activation="relu")(dense)
dense=Dense(10, activation="softmax")(dense)


model = Model(inputs=input1,outputs=dense)

model.summary()

#트레이닝

model.compile(loss="categorical_crossentropy", optimizer="rmsprop",metrics=["accuracy"])
model.fit(x_train,y_train,batch_size=256,epochs=20,validation_split=0.1)

#테스트

loss,acc = model.evaluate(x_test,y_test,batch_size=100)

y_pre=model.predict(x_test)

y_test=np.argmax(y_test,axis=-1)
y_pre=np.argmax(y_pre,axis=-1)
print(__file__)
print(f"loss:{loss}")
print(f"acc:{acc}")
# print(f"x_test.shape:{x_test.shape}")
# print(f"y_pre.shape:{y_pre.shape}")
print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")

'''
keras66_fashion_rnn
loss:0.33359392613172534
acc:0.8902999758720398
y_test[0:20]:[9 2 1 1 6 1 4 6 5 7 4 5 7 3 4 1 2 4 8 0]
y_pre[0:20]:[9 2 1 1 6 1 4 6 5 7 4 5 8 3 4 1 2 4 8 0]
'''