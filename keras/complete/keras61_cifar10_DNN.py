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
dense=Dense(2000,activation="relu")(input1)
dense=Dense(2000,activation="relu")(dense)

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

#epochs=20,layer=1,node=1000
'''
keras61_cifar10_DNN
loss:1.4362430655956269
acc:0.4984000027179718
y_test[0:20]:[3 8 8 0 6 6 1 6 3 1 0 9 5 7 9 8 5 7 8 6]
y_pre[0:20]:[3 8 8 7 4 6 3 4 3 1 8 9 5 2 1 8 3 9 8 6]
'''
#epochs=20,layer=1,node=2000
'''
keras61_cifar10_DNN
loss:1.45334357380867
acc:0.49470001459121704
y_test[0:20]:[3 8 8 0 6 6 1 6 3 1 0 9 5 7 9 8 5 7 8 6]
y_pre[0:20]:[3 1 0 0 4 6 1 4 5 1 8 9 5 7 1 8 5 3 8 6]
'''

#epochs=20,layer=1,node=2000
'''
keras61_cifar10_DNN
loss:1.45334357380867
acc:0.49470001459121704
y_test[0:20]:[3 8 8 0 6 6 1 6 3 1 0 9 5 7 9 8 5 7 8 6]
y_pre[0:20]:[3 1 0 0 4 6 1 4 5 1 8 9 5 7 1 8 5 3 8 6]
'''
#epochs=20,layer=2,node=2000+2000
'''
keras61_cifar10_DNN
loss:1.5295091593265533
acc:0.48750001192092896
y_test[0:20]:[3 8 8 0 6 6 1 6 3 1 0 9 5 7 9 8 5 7 8 6]
y_pre[0:20]:[3 8 8 8 4 6 1 6 2 1 0 9 3 7 1 8 7 4 8 6]
'''