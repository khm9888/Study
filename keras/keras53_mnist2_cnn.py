import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist




(x_train,y_train),(x_test,y_test) = mnist.load_data()

print(f"type(x_train[0]):{type(x_train[0])}")
print(f"x.shape:{x_train.shape}")

print(f"type(x_train):{type(x_train)}")

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

from keras.utils import np_utils
# enc = OneHotEncoder()

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)
# y_train = y_train.reshape(-1,28,28,1)
# y_test = y_test.reshape(-1,28,28,1)

x_train= x_train /255
# y_train= y_train /255

x_test= x_test /255
# y_test= y_test /255
# print(f"x_train:{x_train}")


#모델구성
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten

model= Sequential()

model.add(Conv2D(300,(4,4),input_shape=(28,28,1)))
model.add(Conv2D(300,(4,4),input_shape=(28,28,1)))
# model.add(Conv2D(100,(3,3)))
# model.add(Conv2D(100,(4,4)))


model.add(MaxPool2D(pool_size=(4,4)))
model.add(Flatten())
model.add(Dense(10,activation="softmax"))

model.summary()

#훈련


print("-"*20+str(batch)+"-"*20)
model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["acc"])
model.fit(x_train,y_train,epochs=1,batch_size=batch,validation_split=0.1)

#테스트

loss,acc = model.evaluate(x_test,y_test,batch_size=batch)
y_pre=model.predict(x_test)

y_test=np.argmax(y_test[0:10],axis=1)
y_pre=np.argmax(y_pre[0:10],axis=1)

print(f"loss:{loss}")
print(f"acc:{acc}")

print(f"y_test[0:10]:{y_test[0:10]}")
print(f"y_pre[0:10]:{y_pre[0:10]}")
