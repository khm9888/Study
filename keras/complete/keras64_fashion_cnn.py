import numpy as np

from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import Flatten, MaxPool2D, Input
from keras.datasets import cifar10

#데이터구성
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train[0])
print('y_train[0] : ', y_train[0])

print(f"x_train.shape:{x_train.shape}")
print(f"y_train.shape:{y_train.shape}")


x_train=x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])
x_test=x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

x_train=scaler.fit_transform(x_train)#scaler를 통해서 255로 나눔
x_test=scaler.transform(x_test)

#y값에 np_utils.to_categorical()

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#다시 4차원으로 변경

x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)
#모델

model= Sequential()

# from keras.callbacks import EarlyStopping



model.add(Conv2D(40,(2,2),input_shape=(28,28,1),activation="relu",padding="same"))
model.add(Conv2D(40,(2,2),activation="relu",padding="same"))
# model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(40,(2,2),activation="relu",padding="same"))
model.add(Conv2D(40,(2,2),activation="relu",padding="same"))
# model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(40,(2,2),activation="relu",padding="same"))
model.add(Conv2D(40,(2,2),activation="relu",padding="same"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(10,activation="softmax"))

model.summary()

#트레이닝

model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])
model.fit(x_train,y_train,batch_size=100,epochs=18,validation_split=0.1)

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
keras64_fashion_cnn
loss:0.5751901645958424
acc:0.9017999768257141
y_test[0:20]:[9 2 1 1 6 1 4 6 5 7 4 5 7 3 4 1 2 4 8 0]
y_pre[0:20]:[9 2 1 1 6 1 2 6 5 7 4 5 5 3 4 1 6 6 8 0]
'''