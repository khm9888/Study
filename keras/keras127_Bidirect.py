from keras.datasets import imdb
import numpy as np
import pandas as pd

# 1. 데이터 입력
# print(type(reuters))
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=2000)

print(x_train[0])
print(y_train[0])

print("len(x_train[0])")
print(len(x_train[0]))

#
category = np.max(y_train)+1
print("카테고리")
print(category)

#y의 종류
y_distribution = np.unique(y_train)
print("y_distribution")
print(y_distribution)

y_train_pd = pd.DataFrame(y_train)
print("y_train_pd")
print(y_train_pd[0])
bbb = y_train_pd.groupby(0).count()
# bbb = y_train_pd.groupby(0).count()
print("bbb")
print(bbb)

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

# x_train = pad_sequences(x_train)#truncating
x_train = pad_sequences(x_train,maxlen=111,padding="pre")#truncating
x_test = pad_sequences(x_test,maxlen=111,padding="pre")#truncating

# print(len(x_train[1]))
# print(len(x_train[-1]))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape,y_train.shape)

from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM,Flatten,MaxPool1D,Conv1D,Bidirectional
import matplotlib.pyplot as plt


model = Sequential()
# model.add(Embedding(x_문장개수(word_size),100,input_length=(x_의 단의 개수, 위에서 maxlen)))#통상적인 값

model.add(Embedding(1000,100,input_length=111))
model.add(Conv1D(64, 5, padding='same', activation='relu', strides=1))
# model.add(MaxPool1D(pool_size=4))
# model.add(LSTM(100,activation="relu"))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(39,activation="relu"))
model.add(Dense(2,activation="sigmoid"))

model.summary()


# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
# history = model.fit(x_train,y_train,epochs=15,batch_size=100,validation_split=0.3)

# acc = model.evaluate(x_test,y_test,batch_size=100)[1]
# print("acc")
# print(acc)

# y_pre = model.predict(x_test)

# y_pre=np.argmax(y_pre,axis=1)
# y_test=np.argmax(y_test,axis=1)

# print("y_test[0:20]")
# print(y_test[0:20])
# print("y_train[0:20]")
# print(y_train[0:20])

# y_val_loss = history.history['val_loss'] #AttributeError: 'History' object has no attribute 'histroy'
# y_loss = history.history['loss']

# plt.plot(y_val_loss,marker=".",c="red",label="valid")
# plt.plot(y_loss,marker=".",c="blue",label="train")
# plt.legend(loc="upper right")
# plt.grid()
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.show()


# ######################groupby 사용법 숙지
# #########################1.imdb 검색해서 데이터 확인
# #2.word_size 가장 좋은 수치 확인할 것