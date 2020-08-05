from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import random

def autoencoder(hidden_layer_size=32):#pca 그 순서를 hidden에다가 넣으라구..?

    model = Sequential()
    model.add(Dense(units=hidden_layer_size,activation="relu",input_shape=(784,)))
    model.add(Dense(units=784,activation="sigmoid"))

    return model

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

model_1 = autoencoder(1)
model_2 = autoencoder(2)
model_4 = autoencoder(4)
model_8 = autoencoder(8)
model_16 = autoencoder(16)
model_32 = autoencoder(32)

# model.compile(loss="mse",optimizer="adam",metrics=["acc"])#0.009936471596360207
# # model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["acc"])#0.09251450731754303
# model.fit(x_train,x_train,epochs=10,validation_split=0.2)


model_1.compile(loss="mse",optimizer="adam",metrics=["acc"])
model_2.compile(loss="mse",optimizer="adam",metrics=["acc"])
model_4.compile(loss="mse",optimizer="adam",metrics=["acc"])
model_8.compile(loss="mse",optimizer="adam",metrics=["acc"])
model_16.compile(loss="mse",optimizer="adam",metrics=["acc"])
model_32.compile(loss="mse",optimizer="adam",metrics=["acc"])

model_1.fit(x_train,x_train,epochs=10,validation_split=0.2)
model_2.fit(x_train,x_train,epochs=10,validation_split=0.2)
model_4.fit(x_train,x_train,epochs=10,validation_split=0.2)
model_8.fit(x_train,x_train,epochs=10,validation_split=0.2)
model_16.fit(x_train,x_train,epochs=10,validation_split=0.2)
model_32.fit(x_train,x_train,epochs=10,validation_split=0.2)


# loss,acc=model.evaluate(x_test,x_test)

pre_1 = model_1.predict(x_test)
pre_2 = model_2.predict(x_test)
pre_4 = model_4.predict(x_test)
pre_8 = model_8.predict(x_test)
pre_16 = model_16.predict(x_test)
pre_32 = model_32.predict(x_test)

# print(loss,acc)

fig, axies =plt.subplots(7,5,figsize=(15,15))

random_images = random.sample(range(pre_1.shape[0]),5)
outputs = [x_test,pre_1,pre_2,pre_4,pre_8,pre_16,pre_32]


for row_num,row in enumerate(axies):
    for col_num,ax in enumerate(row):
        ax.imshow(outputs[row_num][random_images[col_num]].reshape(28,28),cmap="gray")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()