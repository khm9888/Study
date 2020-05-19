#1.데이터
import numpy as np

x_train=np.array([i for i in range(1,11)])
y_train=np.array([i for i in range(1,11)])



x_test=np.array([i for i in range(11,16)])
y_test=np.array([i for i in range(11,16)])




#2.모델 구성
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(5,input_dim=1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))


#3.훈련
model.compile(loss="mse",optimizer='adam',metrics=['mse'])
model.fit(x_train,y_train,epochs=500,batch_size=1,validation_data=(y_train,y_train))

#4.평가 및 예측
loss,mse=model.evaluate(x_test,y_test,batch_size=1)

print(f"loss : {loss}")
print(f"mse : {mse}")

output=model.predict(x_test)
print(f"output : \n {output}")

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
