import numpy as np

x=np.array([range(1,101),range(311,411),range(100)])
y=np.array(range(711,811))

x=np.transpose(x)
# y=np.transpose(y)

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split as tts

x_train,x_test,y_train,y_test=tts(x,y,train_size=0.8)
# x_train,x_valid,y_train,y_valid=tts(x_train,y_train,train_size=0.75)
# 여기서 나누지 않고 <3단계 훈련> 과정에 나눌 예정입니다.

#2. 모델 구성

from keras.models import Model,Sequential
from keras.layers import Input,Dense

input1 = Input(shape=(3,))#input
dense1= Dense(5, activation="relu")(input1)
dense2= Dense(4)(dense1)
output1= Dense(1)(dense2)#output

model=Model(inputs=input1,outputs=output1)

# model=Sequential()

# model.add(Dense(5,activation="relu",input_shape=(2,)))
# model.add(Dense(4))
# model.add(Dense(2))

model.summary()

#3. 훈련
model.compile(loss="mse",optimizer="adam",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=100,#validation_data=(x_valid,y_valid))
          validation_split=0.25)

#4. 평가 예측

mse,acc=model.evaluate(x_test,y_test)

print(f"mse:{mse}")
print(f"acc:{acc}")

y_predict=model.predict(x_test)

from sklearn.metrics import mean_squared_error as mse, r2_score

def rmse(y_test,y_predict):
    return np.sqrt(mse(y_test,y_predict))

print(f"rmse:{rmse(y_test,y_predict)}")
print(f"r2:{r2_score(y_test,y_predict)}")



