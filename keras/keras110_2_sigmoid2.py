

### 1. 데이터
import numpy as np

x_train = np.arange(1,10001,1)
y_train = np.array([1,0]*5000)

from keras.utils.np_utils import to_categorical
### 2. 모델
from keras.models import Sequential, Model
from keras.layers import Dense, Input

y_train =to_categorical(y_train)

model = Sequential()

model.add(Dense(1000,activation="elu",input_shape=(1,)))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(2,activation="sigmoid"))


### 3. 실행, 훈련
model.compile(loss = ['binary_crossentropy'], optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=10, batch_size=1,validation_split=0.3)


### 4. 평가, 예측
loss = model.evaluate(x_train, y_train )
print('loss :', loss)

x_pred = np.array([10001, 10002, 10003, 10004])

y_pred = model.predict(x_pred)
y_pred = np.argmax(y_pred,axis=1)
print(y_pred)