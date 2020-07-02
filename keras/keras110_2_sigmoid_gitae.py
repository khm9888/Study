import numpy as np
from keras.models import Sequential
from keras.layers import Dense

x = np.array(range(1,11))
y = np.array([1,0,1,0,1,0,1,0,1,0])

model = Sequential()
model.add(Dense(200,input_dim = 1, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(1, activation='sigmoid'))
## 마지막 activation을 sigmoid같은 0~1사이가 나오는 것을 주어야 한다.

model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['acc'])
## binary_crossentropy == 이진분류의 loss값은 이것 하나밖에 없음

model.fit(x,y, epochs = 1000, batch_size= 32)

loss, acc = model.evaluate(x,y, batch_size=1)
print('loss :',loss)
print('acc :',acc)

y_pred = model.predict([11,12,13,14])
print('y_pred :', y_pred)
print('y_pred :', np.round(y_pred))


##과제 1
## 최종 출력값 0 or 1 만들기