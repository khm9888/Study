from keras.models import Sequential
from keras.layers import Dense,Conv2D,UpSampling2D,MaxPooling2D,Conv2DTranspose
import numpy as np

def autoencoder():#same #0.00982040184289217

    model = Sequential()
    model.add(Conv2D(8,kernel_size=(3,3), activation="relu",input_shape=(28,28,1),padding="same"))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(4,kernel_size=(3,3), activation="relu",padding="same"))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(2,kernel_size=(3,3), activation="relu",padding="same"))

    model.add(Conv2DTranspose(2,kernel_size=(3,3), activation="relu",padding="same"))
    model.add(UpSampling2D((2, 2)))    
    model.add(Conv2DTranspose(4,kernel_size=(3,3), activation="relu",padding="same"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(8,kernel_size=(3,3), activation="relu",padding="same"))
    model.add(Conv2DTranspose(1,kernel_size=(3,3), activation="sigmoid",padding="same"))
    return model

# def autoencoder():#valid #0.022217241019010545

#     model = Sequential()
#     model.add(Conv2D(8,kernel_size=(3,3), activation="relu",input_shape=(28,28,1),padding="same"))
#     model.add(MaxPooling2D((2, 2), padding='valid'))
#     model.add(Conv2D(4,kernel_size=(3,3), activation="relu",padding="valid"))
#     model.add(MaxPooling2D((2, 2), padding='valid'))
#     model.add(Conv2D(2,kernel_size=(3,3), activation="relu",padding="valid"))

#     model.add(Conv2DTranspose(2,kernel_size=(3,3), activation="relu",padding="valid"))
#     model.add(UpSampling2D((2, 2)))    
#     model.add(Conv2DTranspose(4,kernel_size=(3,3), activation="relu",padding="valid"))
#     model.add(UpSampling2D((2, 2)))
#     model.add(Conv2DTranspose(8,kernel_size=(3,3), activation="relu",padding="same"))
#     model.add(Conv2DTranspose(1,kernel_size=(3,3), activation="sigmoid",padding="same"))
#     return model


from keras.datasets import mnist

train_set,test_set = mnist.load_data()
x_train,y_train = train_set
x_test,y_test = test_set

x_train = x_train.reshape(-1,28*28)
x_test = x_test.reshape(-1,28*28)

from sklearn.preprocessing import MinMaxScaler

enc = MinMaxScaler()

x_train = enc.fit_transform(x_train)
x_test = enc.fit_transform(x_test)


x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)


x_train_noised = x_train+np.random.normal(0,0.1,size=x_train.shape)
x_test_noised = x_test+np.random.normal(0,0.1,size=x_test.shape)

x_train_noised=np.clip(x_train_noised,0,1)
x_test_noised=np.clip(x_test_noised,0,1)

#2단계 모델구성
model = autoencoder()

# model.add(Dense(10,activation="so"))

model.compile(loss="mse",optimizer="adam")#0.009936471596360207
# model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["acc"])#0.09251450731754303
model.fit(x_train_noised,x_train_noised,epochs=10,validation_split=0.2)

acc=model.evaluate(x_test_noised,x_test_noised)
pred = model.predict(x_test_noised)

import matplotlib.pyplot as plt
import random

print(loss,acc)

fig, ((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10),(ax11,ax12,ax13,ax14,ax15))=plt.subplots(3,5,figsize=(20,7))

#이미지 무작위 다섯개 픽
random_images = random.sample(range(pred.shape[0]),5)


#원본을 처음에
for i,ax in enumerate((ax1,ax2,ax3,ax4,ax5)):
    ax.imshow(x_test[random_images[i]].reshape(28,28),cmap="gray")
    if i==0:
        ax.set_ylabel("input",size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

#원본을 처음에
for i,ax in enumerate((ax6,ax7,ax8,ax9,ax10)):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28),cmap="gray")
    if i==0:
        ax.set_ylabel("input_noised",size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i,ax in enumerate((ax11,ax12,ax13,ax14,ax15)):
    ax.imshow(pred[random_images[i]].reshape(28,28),cmap="gray")
    if i==0:
        ax.set_ylabel("output",size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()