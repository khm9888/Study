import numpy as np

x=np.array(range(1,101))
y=np.array(range(1,101))

from sklearn.model_selection import train_test_split as tts

x_train,x_test,y_train,y_test=tts(x,y,train_size=0.6,shuffle=False)
x_valid,x_test,y_valid,y_test=tts(x_test,y_test,train_size=0.5,shuffle=False)

from keras.models import Model
#model 클래스를 추가합니다.
from keras.layers import Dense,Input
#Dense, Input 클래스를 추가

input1 = Input(shape=(1,))#입력값
dense1= Dense(5,activation="relu")(input1)
dense2= Dense(3)(dense1)
dense3= Dense(4)(dense2)
output1=Dense(1)(dense3)#출력값

#모델에서 레이어가 각각 함수의 형태로 각각 매칭 시켜줍니다.

model=Model(inputs=input1,outputs=output1)
#입력레이어와 출력레이어도 변수로 넣어줍니다.

model.summary()
#완성된 레이어에 대한 요약 출력합니다.

model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.fit(x_train,y_train,epochs=100,validation_data=(x_valid,y_valid))

loss,acc=model.evaluate(x_test,y_test)

print(f"loss : {loss}")
print(f"acc : {acc}")

y_predict=model.predict(x_test)
print("y_predict : \n {y_predict}")

print(f"y_predict : {y_predict}")

from sklearn.metrics import mean_squared_error as mse,r2_score

def rmse(y_test,y_predict):
    return np.square(mse(y_test,y_predict))


print(y_test)
print(y_predict)

print(f"RMSE : {rmse(y_test,y_predict)}")

print(f"R2 : {r2_score(y_test,y_predict)}")

