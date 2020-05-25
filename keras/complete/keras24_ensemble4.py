#1.데이터 수집
import numpy as np

x1=np.array([range(1,101),range(311,411)])

y1=np.array([range(711,811),range(711,811)])

y2=np.array([range(101,201),range(411,511)])

x1=np.transpose(x1)
y1=np.transpose(y1)
y2=np.transpose(y2)

from sklearn.model_selection import train_test_split as tts

x1_train,x1_test,y1_train,y1_test,y2_train,y2_test=tts(x1,y1,y2,train_size=0.8,shuffle=False)

#모델 구성

from keras.models import Model
from keras.layers import Input, Dense

input1=Input(shape=(2,))
dense1_1=Dense(10,activation="relu")(input1)
dense1_1=Dense(10)(dense1_1)
dense1_1=Dense(100)(dense1_1)
dense1_1=Dense(10)(dense1_1)
dense1_1=Dense(2)(dense1_1)

dense2_1=Dense(10,activation="relu")(dense1_1)
dense2_1=Dense(10)(dense2_1)
dense2_1=Dense(100)(dense2_1)
dense2_1=Dense(10)(dense2_1)
dense2_1=Dense(2)(dense2_1)

dense2_2=Dense(10,activation="relu")(dense1_1)
dense2_2=Dense(10)(dense2_2)
dense2_2=Dense(10)(dense2_2)
dense2_2=Dense(100)(dense2_2)
dense2_2=Dense(10)(dense2_2)
dense2_2=Dense(2)(dense2_2)

model=Model(inputs=input1,outputs=[dense2_1,dense2_2])

#트레이닝

model.compile(loss="mse",optimizer="adam",metrics=["accuracy"])
model.fit(x1_train,[y1_train,y2_train],epochs=100,batch_size=1,validation_split=0.2,verbose=0)

#테스트

values=model.evaluate(x1_test,[y1_test,y2_test],batch_size=1)

#print(model.metrics_names)

for i in range(1,len(values)//2+1):
    print(f"layer{i}`s loss: {values[i]}")
    print(f"layer{i}`s acc: {values[i+2]}")
    

x_pre=np.array([range(101,121),range(411,431)])

x_pre=np.transpose(x_pre)

y_pre=model.predict(x_pre)

for i in range(2):
    print(f"predict({i}) :")
    if i==0:
        for j in range(len(y_pre[i])):
            print(f"test: {y1_test[j]} predict : {y_pre[i][j]}")
    else:
        for j in range(len(y_pre[i])):
            print(f"test: {y2_test[j]} predict : {y_pre[i][j]}")        

from sklearn.metrics import r2_score,mean_squared_error as mse

def rmse(y_test_list,y_pre_list):
    sum_test=0
    sum_pre=0
    for i in range(len(y_test_list)):
        sum_test+=y_test_list[i]
        sum_pre+=y_pre_list[i]
    sum_test//=len(y_test_list)
    sum_pre//=len(y_test_list)
    return np.sqrt(mse(sum_test,sum_pre))

def r2(y_test_list,y_pre_list):
    for i in range(len(y_test_list)):
        print(f"r2({i+1}번) : {r2_score(y_test_list[i],y_pre_list[i])}")
    return 0


r2([y1_test,y2_test],y_pre)   

print(f"rmse:{rmse([y1_test,y2_test],y_pre)}")
