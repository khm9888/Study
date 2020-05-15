#RNN

#데이터 수집

# 1~5,2~6,3~7 입력 -> 6,7,8 출력
import numpy as np

x_train=np.array([range(1,6),range(2,7),range(3,8)])
y_train=np.array(range(3,6))

print(x_train.shape)
print(y_train.shape)

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)

print(x_train.shape)

#모델 구성

from keras.models import Sequential
from keras.layers import Dense,SimpleRNN

model=Sequential()

model.add(SimpleRNN(7,input_shape=(5,1),activation="relu"))
model.add(Dense(5))
model.add(Dense(1))

model.summary()

#훈련
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.fit(x_train,y_train,epochs=100)

x_pre=np.array(range(4,9))
y_pre=model.predict(x_pre)

print(y_pre)



