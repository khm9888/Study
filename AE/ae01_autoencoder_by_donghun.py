import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout,Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist


data = mnist.load_data()
(x_train, y_train),(x_test, y_test) = data

# print(x_train.shape())
# print(y_train.shape())
# print(x_test.shape())
# print(y_test.shape())

# 데이터 전처리
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# x_train = x_train.reshape(60000,28,28,1).astype('float32')/255
# x_test = x_test.reshape(10000,28,28,1).astype('float32')/255

x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000,784).astype('float32')/255

# 2. 모델구성
# model = Sequential()
# model.add(Flatten(input_shape=(28,28,1)))
# model.add(Dense(512,activation='relu'))
# model.add(Dense(10,activation='softmax'))
inputs = Input(shape=(784,))
encoded = Dense(32, activation='relu')(inputs)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(inputs=inputs, outputs=decoded)

autoencoder.summary()

# 3. 컴파일(훈련준비),실행(훈련)
els = EarlyStopping(monitor='loss', patience=10, mode='auto')
autoencoder.compile(optimizer='adam',loss = 'binary_crossentropy', metrics = ['acc'])

hist = autoencoder.fit(x_train,x_train,epochs=15,batch_size=256,callbacks=[],verbose=2)

# 4. 평가, 예측
loss,acc = autoencoder.evaluate(x_test,x_test,batch_size=256)

pred = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(pred[i].reshape(28,28))
    plt.gray
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

print(f"loss : {loss}")
print(f"acc : {acc}")