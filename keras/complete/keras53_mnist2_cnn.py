import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist


batch =128

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
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Activation,Dropout

model= Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

#훈련


print("-"*20+str(batch)+"-"*20)
model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["acc"])
model.fit(x_train,y_train,epochs=20,batch_size=batch,validation_split=0.1)

#테스트

loss,acc = model.evaluate(x_test,y_test,batch_size=batch)
y_pre=model.predict(x_test)

y_test=np.argmax(y_test[0:10],axis=1)
y_pre=np.argmax(y_pre[0:10],axis=1)

print(f"loss:{loss}")
print(f"acc:{acc}")

print(f"y_test[0:10]:{y_test[0:10]}")
print(f"y_pre[0:10]:{y_pre[0:10]}")

# loss:0.052733493065834046
# acc:0.984000027179718
# y_test[0:10]:[7 2 1 0 4 1 4 9 5 9]
# y_pre[0:10]:[7 2 1 0 4 1 4 9 5 9]