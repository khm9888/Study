#1.데이터 수집
import numpy as np

x1=np.array([range(1,101),range(101,201),range(201,301)])
x2=np.array([range(1,101),range(101,201),range(201,301)])
y=np.array(range(101,201))

x1=np.transpose(x1)
x2=np.transpose(x2)
# y=np.transpose(y)


from sklearn.model_selection import train_test_split as tts

x1_train,x1_test,y_train,y_test,x2_train,x2_test=tts(x1,y,x2,train_size=0.8)
# x_train,x_valid,y_train,y_valid=tts(x_train,y_train,train_size=0.75)
# 여기서 나누지 않고 <3단계 훈련> 과정에 나눌 예정입니다.

#2. 모델 구성

from keras.models import Model,Sequential
from keras.layers import Input,Dense,Concatenate

input1 = Input(shape=(3,))#input
dense1= Dense(5, activation="relu")(input1)
dense2= Dense(4)(dense1)
output1= Dense(1)(dense2)#output

# model=Model(inputs=input1,outputs=output1)

input2 = Input(shape=(3,))#input
dense3= Dense(5, activation="relu")(input2)
dense4= Dense(4)(dense3)
output2= Dense(1)(dense4)#output

# model=Model(inputs=input2,outputs=output2)

merge1=Concatenate()([output1,output2])

model1=Dense(10)(merge1)
model2=Dense(5)(model1)
output=Dense(1)(model2)

model=Model(inputs=[input1,input2],outputs=output)

model.summary()

#3.훈련
model.compile(loss="mse",optimizer="adam",metrics=["accuracy"])
model.fit([x1_train,x2_train],y_train,epochs=100,validation_split=0.25)

#데이터처리

mse,acc = model.evaluate([x1_test,x2_test],y_test)

print(f"mse:{mse}")
print(f"acc:{acc}")

y_predict=model.predict([x1_test,x2_test])

print(f"y_predict:\n{y_predict}")

for i in range(len(y_predict)):
    print(y_test[i],y_predict[i])


from sklearn.metrics import mean_squared_error as mse, r2_score

def rmse(y_test,y_predict):
    return np.sqrt(mse(y_test,y_predict))

print(f"rmse:{rmse(y_test,y_predict)}")
print(f"r2:{r2_score(y_test,y_predict)}")

