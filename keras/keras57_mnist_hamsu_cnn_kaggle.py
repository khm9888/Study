import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist


# batch=100

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
from keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout,Input



input1=Input(shape=(28,28,1))
conv=Conv2D(40,(2,2),activation="relu",padding="same")(input1)
conv=Conv2D(40,(2,2),activation="relu",padding="same")(conv)
conv=Conv2D(40,(2,2),activation="relu",padding="same")(conv)
conv=Conv2D(40,(2,2),activation="relu",padding="same")(conv)
conv=Conv2D(40,(2,2),activation="relu",padding="same")(conv)
conv=Conv2D(40,(2,2),activation="relu",padding="same")(conv)
max1=MaxPool2D(pool_size=(2,2))(conv)
flat1=Flatten()(max1)
dense=Dense(10,activation="softmax")(flat1)

model=Model(inputs=input1,outputs=dense)


model.summary()

#훈련


print("-"*20+str("start")+"-"*20)
model.compile(loss="categorical_crossentropy", optimizer="rmsprop",metrics=["acc"])
model.fit(x_train,y_train,epochs=10,batch_size=100,validation_split=0.1)

#테스트

loss,acc = model.evaluate(x_test,y_test,batch_size=100)
y_pre=model.predict(x_test)

y_test=np.argmax(y_test[0:10],axis=1)
y_pre=np.argmax(y_pre[0:10],axis=1)


import pandas as pd

y_pre = model.predict(x_test)
y_pre = np.argmax(y_pre, axis = 1)
y_pre = pd.Series(y_pre, name = 'Label')

submit = pd.concat([pd.Series(range(1, 28001), name = 'ImageId'), y_pre], axis = 1)
submit.to_csv("mySubmission_mnist_cnn.csv", index = False)


print(__file__)
print(f"loss:{loss}")
print(f"acc:{acc}")
y_pre = y_pre.values
print(f"y_test[0:10]:{y_test[0:10]}")
print(f"y_pre[0:10]:{y_pre[0:10]}")

'''
epoch:10
d:\Study\keras\keras57_mnist_hamsu_cnn.py
loss:0.042643220503982775
acc:0.9886999726295471
y_test[0:10]:[7 2 1 0 4 1 4 9 5 9]
y_pre[0:10]:[7 2 1 0 4 1 4 9 5 9]
'''