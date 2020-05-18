#RNN

#데이터 1~5,2~6,3~7을 넣으면 6,7,8이 출력되도록 할 것

#데이터 입력
import numpy as np

x_train=np.array([range(1,6),range(2,7),range(3,8)])
y_train=np.array(range(6,9))

x_test=np.array([range(11,16),range(12,17),range(13,18)])
y_test=np.array(range(16,19))

print(x_train.shape)#(3,5)
print(y_train.shape)#(3,)

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)#(3,5,1)

x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)#(3,5,1)

print(x_train.shape)#(3,5,1)
print(y_train.shape)#(3,)

#모델 구성

from keras.models import Model
from keras.layers import Input,Dense,SimpleRNN

input1=Input(shape=(5,1))
dense1=SimpleRNN(5,activation="relu")(input1)
dense2=Dense(5)(dense1)
dense3=Dense(5)(dense2)
dense4=Dense(5)(dense3)
dense5=Dense(5)(dense4)
output1=Dense(1)(dense5)

model=Model(inputs=input1,outputs=output1)

model.summary()

#트레이닝

model.compile(loss="mse", optimizer="adam",metrics=["accuracy"])
model.fit(x_train,y_train,batch_size=1,epochs=100,verbose=1)

#테스트

x_pre=np.array([range(5,10)])
x_pre=x_pre.reshape(x_pre.shape[0],x_pre.shape[1],1)

print(x_pre.shape)

y_pre=model.predict(x_pre)

mse,acc=model.evaluate(x_test,y_test,batch_size=1)

print(f"mse:{mse}, acc:{acc}")

print(f"y_pre:{y_pre}")

from sklearn.metrics import mean_squared_error as mse,r2_score   

def rmse(y_test,y_pre):
    return np.sqrt(mse(y_test,y_pre))



print(f"rmse:{rmse(y_pre,y_pre)}, r2:{r2_score(y_pre,y_pre)}")