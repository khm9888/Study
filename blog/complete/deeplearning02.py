import numpy as np

x_train=np.array([i for i in range(1,11)])
y_train=np.array([i for i in range(1,11)])

x_valid=np.array([i for i in range(21,31)])
y_valid=np.array([i for i in range(21,31)])

x_test=np.array([i for i in range(41,51)])
y_test=np.array([i for i in range(41,51)])

from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(5,input_dim=1))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

model.compile(loss="mse",optimizer="adam",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=100,validation_data=(x_valid,y_valid))

loss,acc=model.evaluate(x_test,y_test)

print(f"loss : {loss}")
print(f"acc : {acc}")

output=model.predict(x_test)#해당값이 y_predict 값이 된다. 
print(f"output: \n{output}")

from sklearn.metrics import mean_squared_error as mse, r2_score 
#mse - 평균 제곱 오차, r2_score - 결정계수

def RMSE(y_test,y_predict):
    return np.sqrt(mse(y_test,y_predict))
#평균 제곱근 오차 함수 정의


print(f"RMSE : {RMSE(y_test,output)}")
#RMSE - 평균 제곱근 오차 출력

r2_y_predict = r2_score(y_test,output)
#매개변수 

print(f"r2_y_predict : {r2_y_predict}")
#결정계수 출력
