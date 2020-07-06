from keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=2000) # , test_split=0.2)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

print(x_train[0])
print(y_train[0])

#list 이므로 shape 안됨
print('len(x_train[0] :', len(x_train[0]))

# y의 카테고리 개수 출력
category = np.max(y_train) + 1
print("카테고리 : ", category)  # 카테고리 46

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)

y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()

print(bbb)
print(bbb.shape)

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen=111, padding='pre')
x_test = pad_sequences(x_test, maxlen=111, padding='pre')

# print(len(x_train[0]))
# print(len(x_train[-1]))

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)

# 모델 구성
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D, Dropout, Activation, MaxPool1D

model = Sequential()
model.add(Embedding(1000, 128, input_length=111))
# model.add(Embedding(2000, 128))

model.add(Conv1D(64, 5, padding='same', activation='relu', strides=1))
model.add(MaxPool1D(pool_size=4))
model.add(LSTM(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1, activation='sigmoid'))

# model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=100, epochs=10, validation_split=0.2)

acc = model.evaluate(x_test, y_test)[1]
print("acc : ", acc)


# 그림
y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker='.', c='red', label='TestSet Loss')
plt.plot(y_loss, marker='.', c='blue', label='TrainSet Loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()




# 1. imdb검색해서 데이터 내용 확인.
# word_size 전체 데이터 부분 변경해서 최상값 확인.
# groupby


