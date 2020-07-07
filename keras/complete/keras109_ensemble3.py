# 20-07-02_27
# 한 개의 모델에서 분류와 회귀를 동시에 나오게 할 수 있을까?
from keras.layers import concatenate

### 1. 데이터
import numpy as np

x1_train = np.array([1,2,3,4,5,6,7,8,9,10])
x2_train = np.array([11,12,13,14,15,16,17,18,19,20])

y1_train = np.array([1,2,3,4,5,6,7,8,9,10])
y2_train = np.array([1,0,1,0,1,0,1,0,1,0])


### 2. 모델
from keras.models import Sequential, Model
from keras.layers import Dense, Input

#input-1
input1 = Input(shape=(1,))
input2 = Input(shape=(1,))
x1 = Dense(128)(input1)
x1 = Dense(64)(x1)
x1 = Dense(32)(x1)

x2 = Dense(128)(input2)
x2 = Dense(64)(x2)
x2 = Dense(32)(x2)

merge  = concatenate([x1,x2])

#output-1
x3 = Dense(32)(merge)
output1 = Dense(1, activation='relu')(x3)

#output-2
x4 = Dense(32)(merge)
x4 = Dense(16)(x4)
output2 = Dense(1, activation='sigmoid')(x4)

model = Model(inputs = [input1,input2], outputs=[output1, output2])
model.summary()


### 3. 실행, 훈련
model.compile(loss = ['mse', 'binary_crossentropy'], optimizer='adam', metrics=['mse', 'acc'],loss_weights=[0.1,0.9])

model.fit([x1_train,x2_train], [y1_train, y2_train], epochs=100, batch_size=1,verbose=0)


### 4. 평가, 예측
loss = model.evaluate([x1_train,x2_train], [y1_train, y2_train])
print('loss :', loss)

x1_pred = np.array([11, 12, 13, 14])
x2_pred = np.array([11, 12, 13, 14])

y_pred = model.predict([x1_pred,x2_pred])
print(y_pred)