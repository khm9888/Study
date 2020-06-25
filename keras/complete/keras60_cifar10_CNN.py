import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout,BatchNormalization
from keras.layers import Flatten, MaxPooling2D, Input
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

#데이터구성
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train[0])
print('y_train[0] : ', y_train[0])

print(f"x_train.shape:{x_train.shape}")
print(f"y_train.shape:{y_train.shape}")


#minmxasclaer 적용
x_train=x_train.reshape(-1,x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
x_test=x_test.reshape(-1,x_test.shape[1]*x_test.shape[2]*x_test.shape[3])

print(f"x_train.shape:{x_train.shape}")
print(f"y_train.shape:{y_train.shape}")


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

x_train=scaler.fit_transform(x_train)#scaler를 통해서 255로 나눔
x_test=scaler.transform(x_test)

#y값에 np_utils.to_categorical()

# # Generator를 생성하세요
# datagen = ImageDataGenerator(zca_whitening=True)

# # 백색화합니다
# datagen.fit(x_train)
# g = datagen.flow(x_train, y_train, shuffle=False)
# x_train, y_train = g.next()

# # 생성한 이미지를 보기 좋게 만듭니다
# x_train *= 127.0 / max(abs(x_train.min()), abs(x_train.max()))
# x_train += 127
# x_train = x_train.astype('uint8')

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#다시 4차원으로 변경

x_train=x_train.reshape(-1,32,32,3)
x_test=x_test.reshape(-1,32,32,3)

#모델

input1=Input(shape=(32,32,3))
conv=Conv2D(30,(2,2),padding="same",activation="relu")(input1)
conv=Conv2D(30,(2,2),padding="same",activation="relu")(conv)
conv=BatchNormalization()(conv)
conv=Conv2D(30,(2,2),padding="same",activation="relu")(conv)
conv=Conv2D(30,(2,2),padding="same",activation="relu")(conv)
conv=BatchNormalization()(conv)
conv=Conv2D(30,(2,2),padding="same",activation="relu")(conv)
max1=MaxPooling2D(pool_size=3)(conv)
flat1=Flatten()(max1)
dense=Dense(10,activation="softmax")(flat1)


model = Model(inputs=input1,outputs=dense)

model.summary()

#트레이닝

model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])
model.fit(x_train,y_train,batch_size=100,epochs=10,validation_split=0.2)

#테스트

loss,acc = model.evaluate(x_test,y_test,batch_size=100)

y_pre=model.predict(x_test)

y_test=np.argmax(y_test,axis=-1)
y_pre=np.argmax(y_pre,axis=-1)


print("keras60_cifar10_CNN")
print(f"loss:{loss}")
print(f"acc:{acc}")
# print(f"x_test.shape:{x_test.shape}")
# print(f"y_pre.shape:{y_pre.shape}")

print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")

#epochs=20
'''
keras60_cifar10_CNN
loss:1.049088060259819
acc:0.6744999885559082
y_test[0:20]:[3 8 8 0 6 6 1 6 3 1 0 9 5 7 9 8 5 7 8 6]
y_pre[0:20]:[5 1 0 0 6 6 1 6 3 1 0 9 5 7 9 8 5 5 8 6]
'''
#epochs=30
'''
loss:1.2873280984163284
acc:0.6582000255584717
y_test[0:20]:[3 8 8 0 6 6 1 6 3 1 0 9 5 7 9 8 5 7 8 6]
y_pre[0:20]:[3 8 8 8 6 6 1 4 3 1 0 9 5 7 9 6 5 5 8 6]

'''
