from keras.datasets import reuters
import numpy as np
import pandas as pd

# 1. 데이터 입력
# print(type(reuters))
(x_train,y_train),(x_test,y_test) = reuters.load_data(num_words=1000,test_split=0.2)

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
print(y_train_pd)
# bbb = y_train_pd.groupby(0)[0].count()
bbb = y_train_pd.groupby(0).count()
print("bbb")
print(bbb)

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

# x_train = pad_sequences(x_train)#truncating
x_train = pad_sequences(x_train,maxlen=100,padding="pre")#truncating
x_test = pad_sequences(x_test,maxlen=100,padding="pre")#truncating

# print(len(x_train[1]))
# print(len(x_train[-1]))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape,y_train.shape)

from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM,Flatten
import matplotlib.pyplot as plt


model = Sequential()
# model.add(Embedding(x_문장개수(word_size),100,input_length=(x_의 단의 개수, 위에서 maxlen)))#통상적인 값

model.add(Embedding(1000,100,input_length=100))
model.add(LSTM(10,activation="relu"))
model.add(Dense(46,activation="softmax"))


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
hist = model.fit(x_train,y_train,epochs=1,batch_size=100,validation_split=0.3)

acc = model.evaluate(x_test,y_test)[1]
print("acc")
print(acc)

y_pre = model.predict(x_test)

y_pre=np.argmax(y_pre,axis=1)
y_test=np.argmax(y_test,axis=1)

print("y_test[0:20]")
print(y_test[0:20])
print("y_train[0:20]")
print(y_train[0:20])

y_val_loss = hist.histroy['val_loss'] #AttributeError: 'History' object has no attribute 'histroy'
y_loss = hist.histroy['loss']

plt.plot(y_val_loss,marker=".",c="red",label="TestSet Loss")
plt.plot(y_val_loss,marker=".",c="red",label="TestSet Loss")
plt.legend(loc="upper right")
plt.grid()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()