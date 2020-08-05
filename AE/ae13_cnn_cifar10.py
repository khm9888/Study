from keras.models import Sequential
from keras.layers import Dense,Conv2D,UpSampling2D,MaxPooling2D,Conv2DTranspose
import numpy as np
from keras.datasets import cifar10

train_set,test_set = cifar10.load_data()
x_train,y_train = train_set
x_test,y_test = test_set

print(x_train.shape)

x_train = x_train.reshape(-1,32*32*3)
x_test = x_test.reshape(-1,32*32*3)

from sklearn.preprocessing import MinMaxScaler

enc = MinMaxScaler()

x_train = enc.fit_transform(x_train)
x_test = enc.fit_transform(x_test)


x_train = x_train.reshape(-1,32,32,3)
x_test = x_test.reshape(-1,32,32,3)


#2단계 모델구성
def autoencoder():#same

    model = Sequential()
    model.add(Conv2D(8,kernel_size=(3,3), activation="relu",input_shape=(32,32,3),padding="same"))#32
    model.add(MaxPooling2D((2, 2), padding='same'))#16
    model.add(Conv2D(4,kernel_size=(3,3), activation="relu",padding="same"))#16
    model.add(MaxPooling2D((2, 2), padding='same'))#8
    model.add(Conv2D(2,kernel_size=(3,3), activation="relu",padding="same"))#8

    model.add(Conv2DTranspose(2,kernel_size=(3,3), activation="relu",padding="same"))#8
    model.add(UpSampling2D((2, 2)))#16
    model.add(Conv2DTranspose(4,kernel_size=(3,3), activation="relu",padding="same"))
    model.add(UpSampling2D((2, 2)))#32
    model.add(Conv2DTranspose(8,kernel_size=(3,3), activation="relu",padding="same"))
    model.add(Conv2DTranspose(3,kernel_size=(3,3), activation="sigmoid",padding="same"))
    return model

model = autoencoder()

model.summary()

model.compile(loss="mse",optimizer="adam")#0.008583589307963848
# model.compile(loss="binary_crossentropy",optimizer="adam")
model.fit(x_train,x_train,epochs=10,validation_split=0.2)

loss=model.evaluate(x_test,x_test)
pred = model.predict(x_test)

import matplotlib.pyplot as plt
import random

print(loss)

fig, ((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10))=plt.subplots(2,5,figsize=(20,7))

#이미지 무작위 다섯개 픽
random_images = random.sample(range(pred.shape[0]),5)


#원본을 처음에
for i,ax in enumerate((ax1,ax2,ax3,ax4,ax5)):
    ax.imshow(x_test[random_images[i]].reshape(32,32,3),cmap="gray")
    if i==0:
        ax.set_ylabel("input",size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i,ax in enumerate((ax6,ax7,ax8,ax9,ax10)):
    ax.imshow(pred[random_images[i]].reshape(32,32,3),cmap="gray")
    if i==0:
        ax.set_ylabel("output",size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()