#1.데이터
import numpy as np

x=np.array([i for i in range(1,11)])
y=np.array([i for i in range(1,11)])
x_validation=np.array([i for i in range(21,31)])
y_validation=np.array([i for i in range(21,31)])
x_test=np.array([i for i in range(11,14)])
y_test=np.array([i for i in range(11,14)])


#2.모델 구성
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(5,input_dim=1))
model.add(Dense(30))
model.add(Dense(50))

model.add(Dense(40))
model.add(Dense(5))

model.add(Dense(1))


#3.훈련
model.compile(loss="mse",optimizer='adam',metrics=['mse'])
model.fit(x,y,epochs=100,validation_data=(x_validation,y_validation))

#4.평가 및 예측
loss,mse=model.evaluate(x_test,y_test)

print(f"loss : {loss}")
print(f"mse : {mse}")

output=model.predict(x_test)
print(f"output : \n {output}")