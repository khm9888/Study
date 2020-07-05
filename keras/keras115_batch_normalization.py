# regularizer

# l1-가중치 절대값의 합

# l2-가중치 제곱의 합

from keras.datasets import cifar10#데이터 구성
from sklearn.preprocessing import MinMaxScaler#scaler 처리
from keras.models import Model#모델
from keras.layers import Dense, Conv2D, MaxPool2D, Input, Flatten#모델
from keras.layers import BatchNormalization,Activation
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.callbacks import EarlyStopping,ModelCheckpoint
import numpy as np
from keras.optimizers import adam
from keras.regularizers import l1,l2,l1_l2

# print(cifar10.load_data().shape)
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

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

#dimension 맞춰주기 (2->4)

x_train=x_train.reshape(-1,32,32,3)
x_test=x_test.reshape(-1,32,32,3)

#y값에 대해서 to_categorical() 해주기

# y_train=np_utils.to_categorical(y_train)
# y_test=np_utils.to_categorical(y_test)

print(f"y_test.shape:{y_test.shape}")


def build_model_cnn(optimizer=adam,node=32,loss="sparse_categorical_crossentropy"):
# def build_model_cnn(optimizer=adam,node=32,loss="categorical_crossentropy"):
    #2) 모델
    input1=Input(shape=(32,32,3))
    conv=BatchNormalization()(input1)
    conv=Activation("relu")(conv)
    conv=Conv2D(node*2,(2,2),padding="same")(conv)
    conv=BatchNormalization()(conv)
    conv=Activation("relu")(conv)
    conv=Conv2D(node*2,(2,2),padding="same")(conv)
    conv=MaxPool2D(pool_size=3)(conv)

    conv=BatchNormalization()(conv)
    conv=Activation("relu")(conv)
    conv=Conv2D(node*4,(2,2),padding="same")(conv)
    conv=BatchNormalization()(conv)
    conv=Activation("relu")(conv)
    conv=Conv2D(node*4,(2,2),padding="same")(conv)
    conv=MaxPool2D(pool_size=3)(conv)

    conv=BatchNormalization()(conv)
    conv=Activation("relu")(conv)
    conv=Conv2D(node*2,(2,2),padding="same")(conv)
    conv=BatchNormalization()(conv)
    conv=Activation("relu")(conv)
    conv=Conv2D(node*2,(2,2),padding="same")(conv)
    max1=MaxPool2D(pool_size=3)(conv)

    flat1 = Flatten()(max1)
    dense = Dense(10,activation="softmax")(flat1)

    model = Model(inputs=input1,outputs=dense)

    model.summary()

    #3) 훈련

    early = EarlyStopping(monitor="loss",patience=10,mode="min")
    filepath="model/{epoch:02d}-{val_loss:.4f}.hdf5"
    modelcheck = ModelCheckpoint(filepath=filepath,monitor="val_loss",save_best_only=1,mode="min")

    model.compile(loss=loss,optimizer="adam",metrics=["acc"])
    hist=model.fit(x_train,y_train,epochs=40,batch_size=100,validation_split=0.3,callbacks=[early,modelcheck])

    return model,hist

model,hist = build_model_cnn()

#4)테스트

loss,acc = model.evaluate(x_test,y_test)
y_pre= model.predict(x_test)

# y_test=np.argmax(y_test, axis=-1)#인코딩 한 것에 대해 디코딩
# y_pre=np.argmax(y_pre, axis=-1)
print(__file__)

print(f"loss:{loss}")
print(f"acc:{acc}")
print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("keras70.cpfar100_cnn_loss")
plt.legend(["loss","val_loss"])
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(hist.history["acc"])
plt.plot(hist.history["val_acc"])
plt.title("keras70.cpfar100_cnn_acc")
plt.grid()
plt.legend(["acc","val_acc"])
plt.show()
