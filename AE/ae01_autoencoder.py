import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist



#데이터 입력

(x_train,y_train),(x_test,y_test) = mnist.load_data()#데이터 가져옴

x_train, x_test = x_train.reshape(-1,28*28),x_test.reshape(-1,28*28)#scale를 하기 위해 reshape

from sklearn.preprocessing import MinMaxScaler

enc = MinMaxScaler()

x_train=enc.fit_transform(x_train)#scaler를 통해서 255로 나눔
x_test=enc.transform(x_test)

print(x_train.shape)

# from tensorflow.keras.utils import np_utils

# y_train=np_utils.to_categorical(y_train)
# y_test=np_utils.to_categorical(y_test)

#모델구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Input


input1 = Input(shape=(28*28,))
dense = Dense(32,activation="relu")(input1)
dense = Dense(784,activation="sigmoid")(dense)

model=Model(inputs=input1,outputs=dense)

model.summary()

#트레이닝

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train,x_train,batch_size=256,epochs=15,validation_split=0.2)

#테스트
loss,acc = model.evaluate(x_test,x_test,batch_size=256)

y_pre=model.predict(x_test)

import matplotlib.pyplot as pyplot
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax2 = plt.subplot(2,n,i+1+n)
    plt.imshow(y_pre[i].reshape(28,28))
    plt.gray()
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

plt.show()



# y_test=np.argmax(x_test,axis=-1)
# y_pre=np.argmax(x_test,axis=-1)


print(__file__)
print(f"loss:{loss}")
print(f"acc:{acc}")
print(f"y_test[0:10]:{y_test[0:10]}")
print(f"y_pre[0:10]:{y_pre[0:10]}")

