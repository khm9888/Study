from keras.datasets import cifar100#데이터 구성
from sklearn.preprocessing import MinMaxScaler#scaler 처리
from keras.models import Model#모델
from keras.layers import Dense, Conv2D, MaxPool2D, Input, Flatten#모델
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.callbacks import EarlyStopping,ModelCheckpoint
import numpy as np

(x_train,y_train),(x_test,y_test)=cifar100.load_data()

#1) 데이터 구성

print(f"x_test.shape:{x_test.shape}")#x_test.shape:(50000, 32, 32, 3)
print(f"y_test.shape:{y_test.shape}")#y_test.shape:(50000, 1)

#dimension 맞춰주기 (4->2)

x_train=x_train.reshape(-1,x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
x_test=x_test.reshape(-1,x_test.shape[1]*x_test.shape[2]*x_test.shape[3])

#scaler 처리

scaler = MinMaxScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# 필요치 않음, dnn은 2차원

# x_train=x_train.reshape(-1,32,32,3)
# x_test=x_test.reshape(-1,32,32,3)

#y값에 대해서 to_categorical() 해주기

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

print(f"y_test.shape:{y_test.shape}")

#2) 모델

input1= Input(shape=(32*32*3,))

dense = Dense(10000,activation="relu")(input1)
dense = Dense(100,activation="softmax")(dense)

model = Model(inputs=input1,outputs=dense)

model.summary()

#3) 훈련

early = EarlyStopping(monitor="loss",patience=10,mode="min")
filepath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"
modelcheck = ModelCheckpoint(filepath=filepath,monitor="val_loss",save_best_only=1,mode="min")

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["acc"])
hist=model.fit(x_train,y_train,epochs=50,batch_size=100,validation_split=0.1,callbacks=[early,modelcheck])

#4)테스트

loss,acc = model.evaluate(x_test,y_test)
y_pre= model.predict(x_test)

y_test=np.argmax(y_test, axis=-1)#인코딩 한 것에 대해 디코딩
y_pre=np.argmax(y_pre, axis=-1)
print(__file__)

print(f"loss:{loss}")
print(f"acc:{acc}")
print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title('keras71.cpfar100_dnn_loss')    
plt.legend(["loss","val_loss"])
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(hist.history["acc"])
plt.plot(hist.history["val_acc"])
plt.title('keras71.cpfar100_dnn_acc')    
plt.grid()
plt.legend(["acc","val_acc"])
plt.show()


#keras71.cpfar100_dnn

#epoch=20

'''
loss:3.405417025756836
acc:0.2248000055551529
y_test[0:20]:[49 33 72 51 71 92 15 14 23  0 71 75 81 69 40 43 92 97 70 53]
y_pre[0:20]:[ 8 42 15 40 71 44 42  7 71 16 87 97 90 69 40 44 70 31 70 82]
'''

#keras71.cpfar100_dnn

#epoch=20