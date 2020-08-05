import numpy as np

import matplotlib.pyplot as plt

from keras.datasets import mnist

#데이터 입력

(x_train,y_train),(x_test,y_test) = mnist.load_data()#데이터 가져옴

x_train, x_test = x_train.reshape(-1,28*28),x_test.reshape(-1,28*28)#scale를 하기 위해 reshape

from sklearn.preprocessing import MinMaxScaler

enc = MinMaxScaler()

x_train=enc.fit_transform(x_train)#scaler를 통해서 255로 나눔
x_test=enc.transform(x_test)

print(x_train.shape)

from keras.utils import np_utils

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

#모델구성

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(3000,input_shape=(28*28,),activation="relu"))
model.add(Dense(10, activation="sigmoid"))

model.summary()

#트레이닝

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train,y_train,batch_size=256,epochs=30,validation_split=0.1)

#테스트

loss,acc = model.evaluate(x_test,y_test,batch_size=256)

y_pre=model.predict(x_test)

y_test=np.argmax(y_test,axis=-1)
y_pre=np.argmax(y_pre,axis=-1)


print(__file__)
print(f"loss:{loss}")
print(f"acc:{acc}")
print(f"y_test[0:10]:{y_test[0:10]}")
print(f"y_pre[0:10]:{y_pre[0:10]}")

'''

d:\Study\keras\keras56_mnist_dnn.py
loss:0.10343475129455328
acc:0.968500018119812
y_test[0:10]:[7 2 1 0 4 1 4 9 5 9]
y_pre[0:10]:[7 2 1 0 4 1 4 9 6 9]
'''
'''
d:\Study\keras\complete\keras56_mnist_dnn.py
loss:0.07444502667345204
acc:0.9822999835014343
y_test[0:10]:[7 2 1 0 4 1 4 9 5 9]
y_pre[0:10]:[7 2 1 0 4 1 4 9 5 9]
'''