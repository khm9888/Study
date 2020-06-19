import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

import matplotlib.pyplot as plt



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
from keras.callbacks import EarlyStopping, ModelCheckpoint
early = EarlyStopping(monitor="loss", patience=1)
modelpath='./model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint=ModelCheckpoint(filepath=modelpath,monitor="val_loss",save_best_only=1,mode="auto")

print("-"*20+str(100)+"-"*20)
model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["acc"])
hist=model.fit(x_train,y_train,epochs=5,batch_size=200,validation_split=0.1,callbacks=[early,checkpoint])

#테스트

loss,acc = model.evaluate(x_test,y_test,batch_size=200)
y_pre=model.predict(x_test)

y_test=np.argmax(y_test[0:10],axis=1)
y_pre=np.argmax(y_pre[0:10],axis=1)

print(f"loss:{loss}")
print(f"acc:{acc}")

print(f"y_test[0:10]:{y_test[0:10]}")
print(f"y_pre[0:10]:{y_pre[0:10]}")


#히스토리

# plt.subplot(2,1,1)

loss=hist.history["loss"]
val_loss=hist.history["val_loss"]
acc=hist.history["acc"]
val_acc=hist.history["val_acc"]



print(f"loss_acc:{loss}")
print(f"val_loss:{val_loss}")
# print(f"acc:{acc}")
# print(f"val_acc:{val_acc}")



plt.figure(figsize=(10, 6)) #가로세로 길이 그래프설정

plt.subplot(2, 1, 1) #두개의 그림을 그림  (2행 1열에 첫번째 그림을 그리겠다)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')  # 에코가x라서 x값은 안넣었는데 만약 들어가면 이런식으로 plt.plot(x, hist.history['loss'])
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss') #label 레전드안에 안써주고 라벨을 따로 붙혀줄수있다
plt.grid() # 모눈종이처럼 가로세로 줄이 그어짐
plt.title('loss')        # 제목
plt.ylabel('loss')        # y축 이름
plt.xlabel('epoch')            # x축 이름
plt.legend(loc = 'upper right')  #plot 순서에따라 맞춰서 기입   #upperright는 legend 위치 명시


plt.subplot(2, 1, 2) #두개의 그림을 그림  (2행 1열에 첫번째 그림을 그리겠다)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid() # 모눈종이처럼 가로세로 줄이 그어짐
plt.title('acc')        # 제목
plt.ylabel('acc')        # y축 이름
plt.xlabel('epoch')            # x축 이름
plt.legend(['acc', 'val_acc'])

plt.show()