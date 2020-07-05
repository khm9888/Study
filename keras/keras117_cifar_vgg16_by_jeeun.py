import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.applications import VGG16
from keras.optimizers import Adam
import matplotlib.pyplot as plt

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape)    # (50000, 32, 32, 3)
# print(x_test.shape)     # (10000, 32, 32, 3)
# print(y_train.shape)    # (50000, 1)
# print(y_test.shape)     # (10000, 1)

#2. 모델구성
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3)) 
# vgg16 = VGG16()   #(None,224,224,3)
model = Sequential()

model.add(vgg16)
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()

#3. 컴파일, 훈련
model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.3)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
# evaluate하면 loss, acc가 출력
print("loss: ", loss)
print("acc: ", acc)

# y_pred = model.predict(x_test)
# print(y_pred)

#5. 시각화
loss = hist.history['loss']
acc = hist.history['acc']
val_loss = hist.history['val_loss']
val_acc = hist.history['val_acc']

plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(loss, marker='.', c='red', label='loss')
plt.plot(val_loss, marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()

plt.subplot(2,1,2)
plt.plot(acc, marker='.', c='red', label='acc')
plt.plot(val_acc, marker='.', c='blue', label='val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend()
plt.show()