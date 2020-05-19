#1.데이터
import numpy as np

x=np.array([i for i in range(1,11)])
y=np.array([i for i in range(1,11)])
x_validation=np.array([i for i in range(21,31)])
y_validation=np.array([i for i in range(21,31)])
x_test=np.array([i for i in range(101,111)])
y_test=np.array([i for i in range(101,111)])


#2.모델 구성
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(5,input_dim=1))
model.add(Dense(3))
model.add(Dense(500))
model.add(Dense(500))
# model.add(Dense(500))
# model.add(Dense(500))
# model.add(Dense(500))
model.add(Dense(1))


#3.훈련
model.compile(loss="mse",optimizer='adam',metrics=['accuracy'])
model.fit(x_test,y_test,epochs=30,validation_data=(x_validation,y_validation))

#4.평가 및 예측
loss,acc=model.evaluate(x,y)

print(f"loss : {loss}")
print(f"acc : {acc}")

output=model.predict(x)
print(f"output : \n {output}")